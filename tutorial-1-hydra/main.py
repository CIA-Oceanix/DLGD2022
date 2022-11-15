import functools
from pathlib import Path
import pandas as pd
import torch
import src


# Training data
train_dl, val_dl = src.dataloaders(
    dataarray=src.load_ssh(), 
    training_periods=[slice('2013-01-01', '2013-09-30')],
    validation_periods=[slice('2012-10-01', '2012-12-31')],
    number_of_past_days=5,
    batch_size=16
)

# Train model
model = src.train(
    model=torch.nn.Conv2d(5, 1, kernel_size=5, padding=2),
    partial_optimizer=functools.partial(torch.optim.Adam, lr=8e-3),
    dataloaders=(train_dl, val_dl),
    n_epochs=50
)

# Generate metrics
xrds, metrics_df = src.forecast_diagnostic(model, val_dl, number_of_forecast_days=5)
print(metrics_df.to_markdown())


# Log results
logdir = Path('logs') / pd.to_datetime('today').strftime('%y-%m-%d--%H-%M-%S')
logdir.mkdir(exist_ok=True, parents=True)

torch.save(model.state_dict(), logdir / 'weights.t')
xrds.to_netcdf(logdir / 'forecast_data.nc')
metrics_df.to_json(logdir / 'metrics.json')

