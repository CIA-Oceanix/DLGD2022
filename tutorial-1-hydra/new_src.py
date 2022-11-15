import torch
import xarray as xr

def load_sst():
    """Return Sea surface temperature xarray dataarray"""
    return xr.open_dataset('NATL60-CJM165_GULFSTREAM_sst_y2013.1y.nc').sst


class MyForecastModel(torch.nn.Module):
    def __init__(self, ninput, nhidden, nlayers, kernel_size=3, residual=False):
        super().__init__()
        self.residual = residual
        in_channel, out_channel = ninput, nhidden
        self.net = torch.nn.Sequential()

        for layer in range(nlayers):
            self.net.add_module(f'conv_{layer}', torch.nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size//2))
            self.net.add_module(f'bn_{layer}',torch.nn.BatchNorm2d(out_channel))
            self.net.add_module(f'act_{layer}',torch.nn.ReLU())
            in_channel = out_channel
        
        self.net.add_module(f'conv_out',torch.nn.Conv2d(in_channel, 1, kernel_size, padding=kernel_size//2))


    def forward(self, x):
        out = self.net(x)
        if self.residual:
            return x[:, -1:] + out
        return out


