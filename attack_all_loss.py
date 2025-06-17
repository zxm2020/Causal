# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlpackage          # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import torch.nn.functional as F
import torch
from scipy.stats import entropy
import torchvision.models as models
import torchvision.transforms as transforms

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode
import numpy as np
from torch import nn
import torch.optim
import torch.nn.functional as F
import scipy.stats as stats

def add_black_square(image):
    # 图像的宽度和高度
    _,_,height, width= image.shape
    # 黑色方块的尺寸
    square_width, square_height = 240, 320
    # 计算黑色方块的起始坐标
    start_x = (width - square_width) // 2
    start_y = (height - square_height) // 2
    black_square = np.zeros((1, 3, square_height, square_width))
    # 在图像上添加黑色方块
    image[:,:,start_y:start_y+square_height, start_x:start_x+square_width] = black_square
    return image
def get_dimensions(lst):
    if isinstance(lst, list):
        return [len(lst)] + get_dimensions(lst[0]) if lst else []
    else:
        return []

def clip_boxes(boxes, shape):
    """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Rescales (xyxy) bounding boxes from img1_shape to img0_shape, optionally using provided `ratio_pad`."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

# 创建一个辅助函数来计算KL散度
def kl_divergence(p, q):
    """
    计算两个概率分布p和q之间的KL散度
    假设p和q是已正则化的概率分布（和为1）
    """
    # 处理零值，以避免数值不稳定
    p = torch.clamp(p, min=1e-10)
    q = torch.clamp(q, min=1e-10)
    return torch.sum(p * torch.log(p / q))

# 加载ResNet-50模型并设置为评估模式
resnet = models.resnet50(pretrained=True)
resnet.eval()  # 评估模式，关闭dropout等

# 用于提取图像的CNN特征
def extract_features(im):
    """
    使用预训练的ResNet-50提取图像的特征
    返回的特征向量大小为(2048,)
    """
    # im = im.unsqueeze(0).cuda()  # 转换为Tensor并添加batch维度
    # with torch.no_grad():  # 不计算梯度

    features = resnet(im)  # 通过网络进行前向传播
    return features.squeeze()




# @smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.5,  # confidence threshold
    iou_thres=0.5,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    """
    Runs YOLOv5 detection inference on various sources like images, videos, directories, streams, etc.

    Args:
        weights (str | Path): Path to the model weights file or a Triton URL. Default is 'yolov5s.pt'.
        source (str | Path): Input source, which can be a file, directory, URL, glob pattern, screen capture, or webcam
            index. Default is 'data/images'.
        data (str | Path): Path to the dataset YAML file. Default is 'data/coco128.yaml'.
        imgsz (tuple[int, int]): Inference image size as a tuple (height, width). Default is (640, 640).
        conf_thres (float): Confidence threshold for detections. Default is 0.25.
        iou_thres (float): Intersection Over Union (IOU) threshold for non-max suppression. Default is 0.45.
        max_det (int): Maximum number of detections per image. Default is 1000.
        device (str): CUDA device identifier (e.g., '0' or '0,1,2,3') or 'cpu'. Default is an empty string, which uses the
            best available device.
        view_img (bool): If True, display inference results using OpenCV. Default is False.
        save_txt (bool): If True, save results in a text file. Default is False.
        save_csv (bool): If True, save results in a CSV file. Default is False.
        save_conf (bool): If True, include confidence scores in the saved results. Default is False.
        save_crop (bool): If True, save cropped prediction boxes. Default is False.
        nosave (bool): If True, do not save inference images or videos. Default is False.
        classes (list[int]): List of class indices to filter detections by. Default is None.
        agnostic_nms (bool): If True, perform class-agnostic non-max suppression. Default is False.
        augment (bool): If True, use augmented inference. Default is False.
        visualize (bool): If True, visualize feature maps. Default is False.
        update (bool): If True, update all models' weights. Default is False.
        project (str | Path): Directory to save results. Default is 'runs/detect'.
        name (str): Name of the current experiment; used to create a subdirectory within 'project'. Default is 'exp'.
        exist_ok (bool): If True, existing directories with the same name are reused instead of being incremented. Default is
            False.
        line_thickness (int): Thickness of bounding box lines in pixels. Default is 3.
        hide_labels (bool): If True, do not display labels on bounding boxes. Default is False.
        hide_conf (bool): If True, do not display confidence scores on bounding boxes. Default is False.
        half (bool): If True, use FP16 half-precision inference. Default is False.
        dnn (bool): If True, use OpenCV DNN backend for ONNX inference. Default is False.
        vid_stride (int): Stride for processing video frames, to skip frames between processing. Default is 1.

    Returns:
        None

    Examples:
        ```python
        from ultralytics import run

        # Run inference on an image
        run(source='data/images/example.jpg', weights='yolov5s.pt', device='0')

        # Run inference on a video with specific confidence threshold
        run(source='data/videos/example.mp4', weights='yolov5s.pt', conf_thres=0.4, device='0')
        ```
    """




    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path_dig, vid_writer_dig = [None] * bs, [None] * bs
    vid_path_phy, vid_writer_phy = [None] * bs, [None] * bs
    vid_path_pro, vid_writer_pro = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    # perturbation = np.random.uniform(-20, 20, (1, 3, 480, 640))


    learning_rate = 0.1
    import matplotlib.pyplot as plt
    import time
    from PIL import Image
    A = []
    # 读取视频B和C的帧
    cap_bg = cv2.VideoCapture("/home/c405/zxm_code/yolov5-master/runs/detect/move/4.mp4")
    cap_ori = cv2.VideoCapture("/home/c405/zxm_code/yolov5-master/runs/detect/ori/phy.mp4")
    ret_B, frame_B = cap_bg.read()
    ret_C, frame_C = cap_ori.read()
    if ret_B:
        frame_B = torch.tensor(frame_B).float().unsqueeze(0).permute(0, 3, 1, 2) # 转为Tensor

        features_B = extract_features(frame_B)  # 提取CNN特征

        # 计算特征空间中的KL散度
        # 归一化特征向量后再计算KL散度
        p_B = F.softmax(torch.tensor(features_B), dim=0)  # 特征A的softmax
    if ret_C:
        frame_C = torch.tensor(frame_C).float().unsqueeze(0).permute(0, 3, 1, 2) # 转为Tensor

        features_C = extract_features(frame_C)  # 提取CNN特征

        # 计算特征空间中的KL散度
        # 归一化特征向量后再计算KL散度
        p_C = F.softmax(torch.tensor(features_C), dim=0)  # 特征A的softmax
    for path, im, im0s, vid_cap, s in dataset:
        # im(1, 3, 480, 640)
        # im0s(1, 480, 640, 3)

        camera_im = im
        perturbation = np.ones((1, 3, 480, 640))*128/255
        for epoch in range(10):

            perturbation = torch.tensor(perturbation, requires_grad=True)

            # print("1",perturbation)

            perturbed_image = torch.tensor(im, dtype=torch.float32).to(perturbation.device) + perturbation




            A.append(perturbed_image)
            A_tensor = torch.stack(A).to(torch.float32)[0]
            features_A = extract_features(A_tensor)

            q_A = F.softmax(features_A, dim=0)
            kl_AB = kl_divergence(q_A, p_B)#bg
            kl_AC = kl_divergence(q_A, p_C)#ori

            im_ori = torch.clamp(perturbed_image, 0, 255)
            im_transposed = np.transpose(im_ori.detach().numpy(), (0, 2, 3, 1))
            im_transformed = im_transposed[:, :, :, [2, 1, 0]]
            im0s_dig = im_transformed

            with dt[0]:
                im_ori = im_ori.to(model.device)
                im_ori = im_ori.half() if model.fp16 else im_ori.float()  # uint8 to fp16/32


                im_ori /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im_ori.shape) == 3:
                    im_ori = im_ori[None]  # expand for batch dim
                if model.xml and im_ori.shape[0] > 1:
                    ims = torch.chunk(im_ori, im_ori.shape[0], 0)


            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                if model.xml and im_ori.shape[0] > 1:
                    pred = None
                    for image in ims:
                        if pred is None:
                            pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                        else:
                            pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)

                    pred = [pred, None]
                else:
                    pred = model(im_ori, augment=augment, visualize=visualize)
                    # print("3", pred)
                    # grad_fn = < ClampBackward1 >
                    #  grad_fn = < DivBackward0 >

            flag_person = ((pred[0][0][:, 5] > 0.5) & (pred[0][0][:, 4] > 0.2)) * 1
            loss_yolo_person = ((flag_person * pred[0][0][:, 4])).sum()

            flag_others = ((pred[0][0][:, 5] < 0.5) & (pred[0][0][:, 4] > 0.8)) * 1
            loss_yolo_others = ((flag_others * pred[0][0][:, 4])).sum()

            loss = -loss_yolo_person + loss_yolo_others

            # print(loss*100)
            loss.requires_grad_(True)

            # 确保在进行反向传播之前清除之前的梯度
            perturbation.grad = None
            # perturbation.grad.zero_()
            # 反向传播
            loss.backward()
            perturbation_grad_all = perturbation.grad.data
            perturbation_grad = torch.zeros((1, 3, 480, 640))

            # 获取分类置信度部分
            confidences = pred[0][0, :, -80:]
            confidences = confidences.clone().cpu().detach().numpy()
            pred_max = pred[0].cpu().detach().numpy()
            # 找到最大置信度的索引
            # print(confidences.shape)
            max_indices = np.argmax(confidences, axis=1)

            # 创建一个布尔掩码，表示满足条件的备选框
            mask = ((max_indices == 0) & (pred_max[0][:, 4] > 0.5))


            # 使用布尔掩码提取满足条件的box
            boxes = pred_max[0][mask, :4]
            if len(boxes) > 0:

                for i in range(len(boxes)):
                    box = xywh2xyxy(boxes[i])

                    perturbation_grad[:, :, int(box[1]):int(box[3]),
                    int(box[0]):int(box[2])] = perturbation_grad_all[:, :, int(box[1]):int(box[3]),
                                               int(box[0]):int(box[2])]

            perturbation = perturbation + learning_rate * perturbation_grad.sign()
            # perturbation = perturbation_grad.sign() * torch.norm(perturbation, p=2).reciprocal()
        with dt[2]:
            camera_im = torch.from_numpy(camera_im).to(model.device)
            camera_im = camera_im.half() if model.fp16 else camera_im.float()  # uint8 to fp16/32
            camera_im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(camera_im.shape) == 3:
                camera_im = camera_im[None]  # expand for batch dim
            if model.xml and camera_im.shape[0] > 1:
                ims = torch.chunk(camera_im, camera_im.shape[0], 0)
            pred_camera = model(camera_im, augment=augment, visualize=visualize)
            conf_thres = 0.5
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            pred_camera = non_max_suppression(pred_camera, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)



        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        # for i, det in enumerate(pred):  # per image
        i = 0
        det = pred[0]
        det_camera = pred_camera[0]
        seen += 1
        if webcam:  # batch_size >= 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            im0 = np.transpose(im_ori.cpu().detach().numpy(), (0, 2, 3, 1))[0] * 255
            im0_camera = im0s[i][:, :, [2, 1, 0]]
            # im0shape = np.array(im0)
            # print(im0shape.shape)(480, 640, 3)
            s += f"{i}: "
        else:
            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
        im0 = np.array(im0)
        im0_camera = np.array(im0_camera)
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # im.jpg
        txt_path = str(save_dir / "labels" / p.stem) + (
            "" if dataset.mode == "image" else f"_{frame}")  # im.txt
        s += "%gx%g " % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        # # 将 im0 转换为连续张量
        # im0 = im0.contiguous()
        # 将图像转换为连续的数组
        im0 = np.ascontiguousarray(im0)
        im0_camera = np.ascontiguousarray(im0_camera)

        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        annotator_camera = Annotator(im0_camera, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = names[c] if hide_conf else f"{names[c]}"
                confidence = float(conf)
                confidence_str = f"{confidence:.2f}"

                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                    annotator.box_label(xyxy, label, color=colors(c, True))
        if len(det_camera):
            # Rescale boxes from img_size to im0 size
            det_camera[:, :4] = scale_boxes(im.shape[2:], det_camera[:, :4], im0_camera.shape).round()

            # Print results
            for c in det_camera[:, 5].unique():
                n = (det_camera[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy_camera, conf_camera, cls_camera in reversed(det_camera):
                c_camera = int(cls_camera)  # integer class
                label_camera = names[c_camera] if hide_conf else f"{names[c_camera]}"
                confidence_camera = float(conf_camera)
                confidence_camera = f"{confidence_camera:.2f}"

                if save_img or save_crop or view_img:  # Add bbox to image
                    c_camera = int(cls_camera)  # integer class
                    label_camera = None if hide_labels else (names[c_camera] if hide_conf else f"{names[c_camera]} {conf_camera:.2f}")
                    annotator_camera.box_label(xyxy_camera, label_camera, color=colors(c_camera, True))

        # Stream results
        im0 = annotator.result()
        im0_camera = annotator_camera.result()
        im0 = cv2.cvtColor(np.array(im0), cv2.COLOR_RGB2BGR)
        im0_camera = cv2.cvtColor(np.array(im0_camera), cv2.COLOR_RGB2BGR)
        im0 = cv2.convertScaleAbs(im0)
        im0_camera = cv2.convertScaleAbs(im0_camera)
        perturbation_output = np.transpose(perturbation.cpu().detach().numpy(), (0, 2, 3, 1))[0]
        perturbation_output = perturbation_output[:, :, ::-1]
        # perturbation_output = cv2.cvtColor(np.array(perturbation_output), cv2.COLOR_RGB2BGR)
        # perturbation_output = cv2.convertScaleAbs(perturbation_output)
        # image = Image.fromarray((im0).astype(np.uint8))
        # image.save('images/output_image.jpg')
        if view_img:
            # if platform.system() == "Linux" and p not in windows:
            #     windows.append(p)
            #     cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            #     cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])

            height, width = perturbation_output.shape[:2]

            # 计算新的宽度和高度，以保持图像的纵横比
            new_width = int(width * 1.5)  # 例如，将宽度增加50%
            new_height = int(height * 1.5)  # 同样地，将高度增加50%

            # 调整图像大小以适应新的宽度和高度
            resized_image = cv2.resize(perturbation_output, (new_width, new_height))

            # 创建一个名为 "Output Image" 的窗口，并允许调整大小
            cv2.namedWindow("Output Image", cv2.WINDOW_NORMAL)

            # 显示调整后的图像
            cv2.imshow("Output Image", resized_image)

            cv2.imshow("dig", im0)
            cv2.imshow("phy", im0_camera)
            cv2.waitKey(1)  # 1 millisecond
        # Save results (image with detections)
        if save_img:
            if dataset.mode == "image":
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path_dig[i] != save_path:  # new video
                    vid_path_dig[i] = save_path
                    if isinstance(vid_writer_dig[i], cv2.VideoWriter):
                        vid_writer_dig[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 5, im0.shape[1], im0.shape[0]
                    save_path_dig = str(Path(str(save_dir / "dig" )).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                    save_path_phy = str(Path(str(save_dir / "phy")).with_suffix(".mp4"))
                    save_path_pro = str(Path(str(save_dir / "pro")).with_suffix(".mp4"))
                    vid_writer_dig[i] = cv2.VideoWriter(save_path_dig, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer_phy[i] = cv2.VideoWriter(save_path_phy, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer_pro[i] = cv2.VideoWriter(save_path_pro, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                im0_8u_dig = cv2.convertScaleAbs(im0/255, alpha=(255.0))
                im0_8u_phy = cv2.convertScaleAbs(im0_camera/255, alpha=(255.0))
                im0_8u_pro = cv2.convertScaleAbs(perturbation_output, alpha=(255.0))

                vid_writer_dig[i].write(im0_8u_dig)
                vid_writer_phy[i].write(im0_8u_phy)
                vid_writer_pro[i].write(im0_8u_pro)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """
    Parse command-line arguments for YOLOv5 detection, allowing custom inference options and model configurations.

    Args:
        --weights (str | list[str], optional): Model path or Triton URL. Defaults to ROOT / 'yolov5s.pt'.
        --source (str, optional): File/dir/URL/glob/screen/0(webcam). Defaults to ROOT / 'data/images'.
        --data (str, optional): Dataset YAML path. Provides dataset configuration information.
        --imgsz (list[int], optional): Inference size (height, width). Defaults to [640].
        --conf-thres (float, optional): Confidence threshold. Defaults to 0.25.
        --iou-thres (float, optional): NMS IoU threshold. Defaults to 0.45.
        --max-det (int, optional): Maximum number of detections per image. Defaults to 1000.
        --device (str, optional): CUDA device, i.e., '0' or '0,1,2,3' or 'cpu'. Defaults to "".
        --view-img (bool, optional): Flag to display results. Defaults to False.
        --save-txt (bool, optional): Flag to save results to *.txt files. Defaults to False.
        --save-csv (bool, optional): Flag to save results in CSV format. Defaults to False.
        --save-conf (bool, optional): Flag to save confidences in labels saved via --save-txt. Defaults to False.
        --save-crop (bool, optional): Flag to save cropped prediction boxes. Defaults to False.
        --nosave (bool, optional): Flag to prevent saving images/videos. Defaults to False.
        --classes (list[int], optional): List of classes to filter results by, e.g., '--classes 0 2 3'. Defaults to None.
        --agnostic-nms (bool, optional): Flag for class-agnostic NMS. Defaults to False.
        --augment (bool, optional): Flag for augmented inference. Defaults to False.
        --visualize (bool, optional): Flag for visualizing features. Defaults to False.
        --update (bool, optional): Flag to update all models in the model directory. Defaults to False.
        --project (str, optional): Directory to save results. Defaults to ROOT / 'runs/detect'.
        --name (str, optional): Sub-directory name for saving results within --project. Defaults to 'exp'.
        --exist-ok (bool, optional): Flag to allow overwriting if the project/name already exists. Defaults to False.
        --line-thickness (int, optional): Thickness (in pixels) of bounding boxes. Defaults to 3.
        --hide-labels (bool, optional): Flag to hide labels in the output. Defaults to False.
        --hide-conf (bool, optional): Flag to hide confidences in the output. Defaults to False.
        --half (bool, optional): Flag to use FP16 half-precision inference. Defaults to False.
        --dnn (bool, optional): Flag to use OpenCV DNN for ONNX inference. Defaults to False.
        --vid-stride (int, optional): Video frame-rate stride, determining the number of frames to skip in between
            consecutive frames. Defaults to 1.

    Returns:
        argparse.Namespace: Parsed command-line arguments as an argparse.Namespace object.

    Example:
        ```python
        from ultralytics import YOLOv5
        args = YOLOv5.parse_opt()
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov3.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "0", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Executes YOLOv5 model inference based on provided command-line arguments, validating dependencies before running.

    Args:
        opt (argparse.Namespace): Command-line arguments for YOLOv5 detection. See function `parse_opt` for details.

    Returns:
        None

    Note:
        This function performs essential pre-execution checks and initiates the YOLOv5 detection process based on user-specified
        options. Refer to the usage guide and examples for more information about different sources and formats at:
        https://github.com/ultralytics/ultralytics

    Example usage:

    ```python
    if __name__ == "__main__":
        opt = parse_opt()
        main(opt)
    ```
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
