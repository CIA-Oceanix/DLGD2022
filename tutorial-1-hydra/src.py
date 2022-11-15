import xarray as xr
import pandas as pd
import scipy.ndimage as ndi
import numpy as np
import tqdm
import torchmetrics
import torch.nn.functional as F

import torch
import itertools as it

import hydra
import hydra.utils


################### dataset utils ###################################
class XrDataset(torch.utils.data.Dataset):
    """
    torch Dataset based on an xarray.DataArray
    """
    def __init__(
            self, da, patch_dims, domain_limits=None, strides=None):
        """
        da: xarray.DataArray
        patch_dims: dict of da dimension to size of a patch 
        domain_limits: dict of da dimension to slices of domain to select for patch extractions
        """
        super().__init__()
        self.da = da.sel(**(domain_limits or {})).transpose('time', 'lat', 'lon')
        self.patch_dims = patch_dims
        self.strides = strides or {}
        da_dims = dict(zip(self.da.dims, self.da.shape))
        self.ds_size = {
            dim: max((da_dims[dim] - patch_dims[dim]) // self.strides.get(dim, 1) + 1, 0)
            for dim in patch_dims
        }
        self._return_coords = False


    def __len__(self):
        size = 1
        for v in self.ds_size.values():
            size *= v
        return size

    def get_coords(self):
        self._return_coords = True
        coords = []
        try:
            for i in range(len(self)):
                coords.append(self[i])
        finally:
            self._return_coords = False
            return coords

    def __getitem__(self, item):
        sl = {dim: slice(self.strides.get(dim, 1) * idx,
                           self.strides.get(dim, 1) * idx + self.patch_dims[dim])
                for dim, idx in zip(self.ds_size.keys(),
                                    np.unravel_index(item, tuple(self.ds_size.values())))}
        if self._return_coords:
            return self.da.isel(**sl).coords.to_dataset()[['time', 'lat', 'lon']]
        full_item = self.da.isel(**sl).data.astype(np.float32)
        return full_item[:-1], full_item[-1:]


class XrConcatDataset(torch.utils.data.ConcatDataset):
    """
    Concatenation of XrDatasets
    """
    def get_coords(self):
        return it.chain(*(ds.get_coords() for ds in self.datasets))


################### DataLoading utils ###################################

def load_ssh():
    """Return Sea surface height xarray dataarray"""
    return (xr.open_dataset('NATL60-CJM165_GULFSTREAM_ssh_y2013.1y.nc', decode_times=False)
    .assign_coords(time=lambda ds: pd.to_datetime(ds.time) + pd.to_timedelta('12H'))).ssh


def dataloaders(dataarray, training_periods, validation_periods, number_of_past_days, batch_size):
    """
    dataarray: xarray dataarray - input data
    training_periods: list of slices of date - for training (e.g: [slice("2012/10/15","2012/11/15")])
    validation_periods: list of slices of date - for validation 
    number_of_past_days: int number of days to use as input for the forecast

    Returns two torch.utils.data.DataLoader for training and validation
    """

    train_ds = XrConcatDataset(
        [XrDataset(dataarray, patch_dims=dict(time=number_of_past_days + 1, lat=201, lon=201),
                   domain_limits=dict(time=period))
        for period in training_periods]
    )
    val_ds = XrConcatDataset(
        [XrDataset(dataarray, patch_dims=dict(time=number_of_past_days + 1, lat=201, lon=201), 
                   domain_limits=dict(time=period))
        for period in validation_periods]
    )
    return [torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1),
            torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1)]

################### Training  ###################################

def train(model, partial_optimizer, dataloaders, n_epochs):
    """
    model: torch.nn.Module with inputs of size (bs, number_of_past_days, 201, 201) and output of size (bs, 1, 201, 201)
    partial_optimizer: torch optimizer constructor that takes model parameters as input
    dataloaders: list of the two training and validation dataloaders
    n_epochs: number of epochs 
    """

    train_dl, val_dl = dataloaders
    train_rmse, val_rmse = torchmetrics.MeanSquaredError(squared=False), torchmetrics.MeanSquaredError(squared=False)
    baseline_val_rmse = torchmetrics.MeanSquaredError(squared=False)
    best_score = np.inf
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    best_model_weights = model.state_dict()
    opt = partial_optimizer(model.parameters())

    pbar = tqdm.tqdm(range(n_epochs))
    for epoch in pbar:
        for x, y in train_dl:
            opt.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = F.mse_loss(y_hat, y)
            loss.backward()
            opt.step()
            train_rmse(y_hat.cpu(), y.cpu())

        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            val_rmse(y_hat.cpu(), y.cpu())
            baseline_val_rmse(x[:, -1:].cpu(), y.cpu())

        val_score = val_rmse.compute()
        if val_score < best_score:
            best_score = val_score
            best_model_weights = model.state_dict()

        pbar.set_description(
            f"Epoch: {epoch} \t - train err: {train_rmse.compute():.3f} - val err: {val_score:.3f} (base err: {baseline_val_rmse.compute():.3f}) (m)")
        train_rmse.reset(), val_rmse.reset(), baseline_val_rmse.reset()

    model.load_state_dict(best_model_weights)
    return model

################### Generate Forecast  ###################################

def generate_forecast(model, dl, number_of_forecast_days=1):
    """
    model: torch.nn.Module 
    dl: dataloader on which to generate the forecast
    number_of_forecast_days: number of days to forecast for a single input (using successive application of the model)

    Returns xarray dataset with dimensions (
        'time',
        'lat',
        'lon',
        'day_plus' -> number of days since last day of input
           )
    and with the variables: [
        'gt',
        'persistence', -> last input day
        'forecast', -> model forecast
    ]
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    coords = dl.dataset.get_coords()

    forecast_xrds = (
        xr.Dataset(coords=xr.concat([c.isel(time=-1) for c in coords], dim='time'))
        .assign(
            gt=lambda ds: (ds.dims, np.full(list(ds.dims.values()), np.nan)),
            persistence=lambda ds: ([*ds.dims, 'day_plus'], np.full([*ds.dims.values(), number_of_forecast_days], np.nan)),
            forecast=lambda ds: ([*ds.dims, 'day_plus'], np.full([*ds.dims.values(), number_of_forecast_days], np.nan)),
        ))
    
    with torch.no_grad():
        bs, be = 0, 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            be = bs + x.shape[0]
            batch_slice = dict(time=slice(bs, be))

            forecast_xrds['gt'][batch_slice] = y.cpu().squeeze(dim=1).numpy() 
            for day in range(number_of_forecast_days):
                if x.shape[1] >= (day+1):
                    forecast_xrds['persistence'][{**batch_slice, 'day_plus': day}] = x[:, -(day+1)].cpu().numpy() 

            for day in range(number_of_forecast_days):
                y_hat = model(x)
                forecast_slice = dict(time=slice(bs + day, min(be + day, forecast_xrds.dims['time'])))
                if (be + day) > forecast_xrds.dims['time']:
                    y_hat = y_hat[:-1]
                    x = x[:-1]

                forecast_xrds[f'forecast'][{**forecast_slice, 'day_plus': day}] = y_hat.cpu().squeeze(dim=1).numpy() 
                x = torch.cat([x[:, 1:], y_hat], dim=1)
            bs=be
    return forecast_xrds   

def forecast_diagnostic(model, dl, number_of_forecast_days=1):
    """
    model: torch.nn.Module 
    dl: dataloader on which to generate the forecast
    number_of_forecast_days: number of days to forecast for a single input (using successive application of the model)

    Returns xarray dataset with dimensions (
        'time',
        'lat',
        'lon',
        'day_plus' -> number of days since last day of input
           )
    and with the variables: [
        'gt',
        'persistence', -> last input day
        'forecast', -> model forecast
    ]
    and a pandas.DataFrame with reconstrution error
    """
    forecast_xrds = generate_forecast(model, dl, number_of_forecast_days)
    metrics_df = (
        forecast_xrds
        .pipe(lambda ds: ds-ds.gt).drop('gt')
        .pipe(lambda ds: np.sqrt((ds**2).mean(dim=['time', 'lat', 'lon'])))
        .to_dataframe()
        .rename(lambda c: f'{c} error (cm)', axis=1)
        .assign(**{'improvement (%)':lambda df:  1 - df['forecast error (cm)'] / df['persistence error (cm)']})
        .applymap(lambda i: f'{100*i:.2f}')
        .rename(lambda i:f'day + {i+1}', axis=0).T
    )
    return forecast_xrds, metrics_df

