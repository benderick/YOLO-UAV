# set working directory, get file path
import os
from pathlib import Path
file_path = Path(__file__).resolve()
ROOT = file_path.parent.parent.parent
os.chdir(file_path.parent)

# set cfg
project_name = file_path.parent.parent.name
run_name = file_path.parent.name
model_cfg = "yolo11s.yaml"
data_cfg = ROOT / "cfg/datasets/VisDrone.yaml"
set_cfg = ROOT / "cfg/setting.yaml"
logdir = ROOT.parent / "tmp/yolo-runs"
# -------------------------------------------------

from ultralytics import YOLO

model = YOLO(model_cfg.name)

model.train(
    project=f"{logdir.name}/{project_name}",
    name=run_name,
    data=data_cfg.name, 
    cfg=set_cfg.name,
    epochs=300,
    device="0,1,2,3",
    batch=64
    )