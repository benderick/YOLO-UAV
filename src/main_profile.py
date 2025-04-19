import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # choose your yaml file
    model = YOLO('/icislab/volume3/benderick/futurama/YOLO-UAV/configs/model/mixup/yolo11-EnhancedMAB-MAB-ADown.yaml')
    model.info(detailed=True)
    try:
        model.profile(imgsz=[640, 640])
    except Exception as e:
        print(e)
        pass
    model.fuse()