from ultralytics import YOLO

model = YOLO("yolo11s.yaml")

model.tune(data='/icislab/volume1/zhangshuo/YOLO-UAV/cfg/VisDrone.yaml', epochs=70,iterations=40,device="0,3,4,5",batch=256,project="YOLO11s-VisDrone-Tune",name="1")