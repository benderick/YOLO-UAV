# set working directory, get root path
from omegaconf import DictConfig
import rootutils
ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import pretty_errors
from rich import print

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import hydra
from ultralytics.utils import oc_to_dict

import ultralytics
from ultralytics import RTDETR

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    ultralytics.settings["datasets_dir"] = cfg.paths.data_dir
    ultralytics.settings["wandb"] = cfg.use_wandb
    model = RTDETR(oc_to_dict(cfg.model))
    model.train(
        project=f"{cfg.paths.output_dir}/{cfg.project_name}",
        name=cfg.run_name,
        data=cfg.data.file, 
        cfg=cfg.setting,
        epochs=cfg.epochs,
        device=cfg.device,
        batch=cfg.batch,
        # fraction=0.1,
        patience=0,
        amp=False,
        logger=str(oc_to_dict(cfg.logger)),
        data_layout=cfg.data_layout,    
    )

if __name__ == "__main__":
    main()