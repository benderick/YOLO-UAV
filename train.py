# set working directory, get root path
import rootutils
ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import pretty_errors
from rich import print

import hydra
from omegaconf import DictConfig

import ultralytics
from ultralytics import YOLO

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    ultralytics.settings["datasets_dir"] = cfg.paths.data_dir
    model = YOLO(cfg.model.cfg_file)
    model.train(
        project=f"{cfg.paths.log_dir}/{cfg.project_name}",
        name=cfg.run_name,
        data=cfg.data.cfg_file, 
        cfg=cfg.mode.cfg_file,
        epochs=cfg.epochs,
        device=cfg.device,
        batch=cfg.batch,
        fraction=0.01
        )

if __name__ == "__main__":
    main()