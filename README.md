# YOLO-UAV
这是为YOLO-UAV准备的实验文件夹。
各个模型都不使用预训练权重。

## 实验配置
|编号|GPU|环境|CUDA|PyTorch|
|---|---|---|---|---|
|188|3090|lightning|11.8||
|190|a30|yolo|11.3||
|186|v100|yolo|11.7|2.0.1|

## 事项列表
1. 基于YOLO11s，在VisDrone上寻找最优超参数
    - epoches=50, iterations=50
    - 使用190跑
    - 参数：
    ```python
    model.tune(data='../cfg/VisDrone.yaml', 
           epochs=50,
           iterations=50,
           device="6,7",
           batch=64, 
           project="../../tmp/yolo-runs/YOLO11s-VisDrone-Tune",name="tune",
           patience=10)
    ```
    - 在cfg文件夹下的VisDrone-Tune.yaml用来覆盖default.yaml
2. YOLO11s，在VisDrone跑300轮

