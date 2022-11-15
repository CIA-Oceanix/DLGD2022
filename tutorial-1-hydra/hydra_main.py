from pathlib import Path
import hydra
from omegaconf import OmegaConf
import torch


@hydra.main(config_path='config', config_name='main', version_base="1.2")
def main(cfg):
    print(f"{ ' Job config ':#^64}")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Training data
    train_dl, val_dl = hydra.utils.call(cfg.data) 
    
    # Training model
    model = hydra.utils.call(cfg.training, dataloaders=(train_dl, val_dl))

    # Compute diagnostics
    xrds, metrics_df = hydra.utils.call(cfg.diagnostic, model=model, dl=val_dl)
    print(metrics_df.to_markdown())
    
    # Logging
    logdir = Path(cfg.logdir)
    logdir.mkdir(exist_ok=True, parents=True)

    torch.save(model.state_dict(), logdir / 'weights.t')
    xrds.to_netcdf(logdir / 'forecast_data.nc')
    metrics_df.to_json(logdir / 'metrics.json')
    (logdir / 'config.yaml').write_text(OmegaConf.to_yaml(cfg))
    (logdir / 'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ =='__main__':
    main()

