from ultralytics import YOLO

model = YOLO("yolo11s.yaml")

model.tune(data='../cfg/VisDrone.yaml', 
           epochs=50,
           iterations=50,
           device="6,7",
           batch=64, 
           project="../../tmp/yolo-runs/YOLO11s-VisDrone-Tune",name="tune",
           patience=10)

# model.train(data='../cfg/VisDrone.yaml', epochs=3,device="6,7",batch=64,project="../../tmp/yolo-runs/YOLO11s-VisDrone-Tune",name="tune")
