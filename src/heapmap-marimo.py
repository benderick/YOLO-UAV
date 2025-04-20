import marimo

__generated_with = "0.12.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 基本""")
    return


@app.cell
def _():
    import warnings

    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    import torch, yaml, cv2, os, shutil
    import numpy as np
    import os
    import copy

    np.random.seed(0)
    from ultralytics import YOLO
    import matplotlib.pyplot as plt
    from tqdm import trange
    from PIL import Image
    from ultralytics.nn.tasks import DetectionModel as Model
    from ultralytics.utils.torch_utils import intersect_dicts
    from ultralytics.utils.ops import xywh2xyxy
    from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
    return (
        ActivationsAndGradients,
        GradCAM,
        GradCAMPlusPlus,
        Image,
        Model,
        XGradCAM,
        YOLO,
        copy,
        cv2,
        intersect_dicts,
        np,
        os,
        plt,
        show_cam_on_image,
        shutil,
        torch,
        trange,
        warnings,
        xywh2xyxy,
        yaml,
    )


@app.cell
def _(cv2, np):
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)
    return (letterbox,)


@app.cell
def _(
    ActivationsAndGradients,
    Model,
    cv2,
    intersect_dicts,
    letterbox,
    np,
    show_cam_on_image,
    torch,
    trange,
    xywh2xyxy,
):
    class yolo_heatmap:
        def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio):
            device = torch.device(device)
            ckpt = torch.load(weight)
            model_names = ckpt['model'].names
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            model = Model(cfg, ch=3, nc=len(model_names)).to(device)
            csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # intersect
            model.load_state_dict(csd, strict=False)  # load
            model.eval()
            print(f'Transferred {len(csd)}/{len(model.state_dict())} items')

            target_layers = [eval(layer)]
            method = eval(method)

            colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
            self.__dict__.update(locals())

        def post_process(self, result):
            logits_ = result[:, 4:]
            boxes_ = result[:, :4]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
                indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

        def draw_detections(self, box, color, name, img):
            xmin, ymin, xmax, ymax = list(map(int, list(box)))
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
            cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                        lineType=cv2.LINE_AA)
            return img

        def __call__(self, img_path):
            # img process
            img = cv2.imread(img_path)
            img = letterbox(img)[0]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.float32(img) / 255.0
            tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

            # init ActivationsAndGradients
            grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)

            # get ActivationsAndResult
            result = grads(tensor)
            activations = grads.activations[0].cpu().detach().numpy()

            # postprocess to yolo output
            post_result, pre_post_boxes, post_boxes = self.post_process(result[0])
            all_maps = []
            for i in trange(int(post_result.size(0) * self.ratio)):
                if float(post_result[i].max()) < self.conf_threshold:
                    break

                self.model.zero_grad()
                # get max probability for this prediction
                if self.backward_type == 'class' or self.backward_type == 'all':
                    score = post_result[i].max()
                    score.backward(retain_graph=True)

                if self.backward_type == 'box' or self.backward_type == 'all':
                    for j in range(4):
                        score = pre_post_boxes[i, j]
                        score.backward(retain_graph=True)

                # process heatmap
                if self.backward_type == 'class':
                    gradients = grads.gradients[0]
                elif self.backward_type == 'box':
                    gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3]
                else:
                    gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3] + \
                                grads.gradients[4]
                b, k, u, v = gradients.size()
                weights = self.method.get_cam_weights(self.method, None, None, None, activations,
                                                      gradients.detach().numpy())
                weights = weights.reshape((b, k, 1, 1))
                saliency_map = np.sum(weights * activations, axis=1)
                saliency_map = np.squeeze(np.maximum(saliency_map, 0))
                saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
                saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
                if (saliency_map_max - saliency_map_min) == 0:
                    continue
                saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

                all_maps.append(saliency_map)
            saliency_map = np.mean(all_maps, axis=0)
            try:
            # add heatmap and box to image
                cam_image = show_cam_on_image(img.copy(), saliency_map, use_rgb=True)
            except:
                cam_image = img
            "不想在图片中绘画出边界框和置信度，注释下面的一行代码即可"
            # cam_image = self.draw_detections(post_boxes[i], self.colors[int(post_result[i, :].argmax())],
            #                                  f'{self.model_names[int(post_result[i, :].argmax())]} {float(post_result[i].max()):.2f}',
            #                                  cam_image)
            # cam_image = Image.fromarray(cam_image)
            # cam_image.show()
            return cam_image
    return (yolo_heatmap,)


@app.cell
def _(control_conf, control_layer1, control_layer2):
    def make_params(weight, cfg,  m=1):
        return {
            'weight': weight,  # 训练出来的权重文件
            'cfg': cfg,
            'device': 'cuda:0',
            'method': 'GradCAM',  # GradCAMPlusPlus, GradCAM, XGradCAM , 使用的热力图库文件不同的效果不一样可以多尝试
            'layer': f"model.model[{control_layer1.value if m==1 else control_layer2.value}]",  # 想要检测的对应层
            'backward_type': 'all',  # class, box, all
            'conf_threshold': control_conf.value,  # 0.6  # 置信度阈值，有的时候你的进度条到一半就停止了就是因为没有高于此值的了
            'ratio': 0.02, # 0.02-0.1
            #'show_result': False, # 不需要绘制结果请设置为False
            #'renormalize': False, # 需要把热力图限制在框内请设置为True(仅对detect,segment,pose有效)
            #'task':'detect', # 任务(detect,segment,pose,obb,classify)
            #'img_size':640, # 图像尺寸
        }
    return (make_params,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 配置""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### 选择模型""")
    return


@app.cell
def _(make_params, yolo_heatmap):
    model1 = yolo_heatmap(**make_params(
        "/icislab/volume3/benderick/futurama/YOLO-UAV/logs/YOLO-UAV/multiruns/2025-04-18_02-04-04/0-yolo11/YOLO-UAV/yolo11/weights/best.pt",
        "/icislab/volume3/benderick/futurama/YOLO-UAV/configs/model/artifact-0/yolo11.yaml",
        1
    ))
    return (model1,)


@app.cell
def _(make_params, yolo_heatmap):
    model2 = yolo_heatmap(**make_params(
        '/icislab/volume3/benderick/futurama/YOLO-UAV/logs/YOLO-UAV/runs/2025-04-19_19-45-37-EAB-1/YOLO-UAV/EAB-1/weights/best.pt',
        "/icislab/volume3/benderick/futurama/YOLO-UAV/configs/model/artifact-0/EAB-1.yaml",
        2
    ))
    return (model2,)


@app.cell
def _():
    imgs_dir = "/icislab/volume3/benderick/futurama/YOLO-UAV/data/VisDrone/sample"
    return (imgs_dir,)


@app.cell(hide_code=True)
def _(imgs_dir, os):
    img_list = os.listdir(imgs_dir)
    return (img_list,)


@app.cell(hide_code=True)
def _(img_list, mo):
    control_img = mo.ui.slider(0, len(img_list)-1, show_value=True, label="选择图片：", value=0)
    control_img
    return (control_img,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""#### 一般看第4层，16层""")
    return


@app.cell(hide_code=True)
def _(mo):
    control_layer1 = mo.ui.slider(1, 23, show_value=True, label="M1:检测第几层：", value=16)
    control_layer1
    return (control_layer1,)


@app.cell(hide_code=True)
def _(mo):
    control_layer2 = mo.ui.slider(1, 23, show_value=True, label="M2:检测第几层：", value=16)
    control_layer2
    return (control_layer2,)


@app.cell(hide_code=True)
def _(mo):
    control_conf = mo.ui.slider(0.1,1,0.1, show_value=True, label="置信度", value=0.9)
    control_conf
    return (control_conf,)


@app.cell
def _(control_img, img_list, imgs_dir, model1):
    img1 = model1(imgs_dir + "/" + img_list[control_img.value])
    return (img1,)


@app.cell
def _(control_img, img_list, imgs_dir, model2):
    img2 = model2(imgs_dir + "/" + img_list[control_img.value])
    return (img2,)


@app.cell(hide_code=True)
def _(img1, img2, plt):
    # 创建一行两列的子图
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))  # 1行2列，设置图像大小

    # 第一个子图
    axs[0].imshow(img1)
    axs[0].axis('off')  # 关闭坐标轴
    axs[0].set_title("1")

    # 第二个子图
    axs[1].imshow(img2)
    axs[1].axis('off')  # 关闭坐标轴
    axs[1].set_title("2")

    # 自动调整布局防止重叠
    plt.tight_layout()

    # 显示图像
    plt.show()
    return axs, fig


if __name__ == "__main__":
    app.run()
