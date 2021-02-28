import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import numpy as np
def is_color_image(image):
    if len(image.shape) == 3 and image.shape[2] >= 3:
        return True
    return False
def compositeDiff(preImage, curImage, nextImage):
    if is_color_image(preImage):
        preImage = cv2.cvtColor(preImage, cv2.COLOR_BGR2GRAY)
    preImage = preImage.astype(np.float32)
    if is_color_image(curImage):
        curImage = cv2.cvtColor(curImage, cv2.COLOR_BGR2GRAY)
    curImage = curImage.astype(np.float32)
    if is_color_image(nextImage):
        nextImage = cv2.cvtColor(nextImage, cv2.COLOR_BGR2GRAY)
    nextImage = nextImage.astype(np.float32)
    diff0 = curImage - preImage
    diff1 = curImage - nextImage
    x = cv2.merge([curImage, diff0, diff1])
    return x
CycleClass = ["Ds","Dd","Inter"]
StdClass = ["Std", "Nstd"]
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    # ********************************* lzz append1 *********************************
    ImgList = []
    LabelCycleList = []
    LabelStdList = []
    ResLabelCycleList = []
    PreImage = None
    CurImage = None
    NextImage = None
    CurImageim0s = None
    NextImageim0s = None
    RealImage = None
    nowcount = 1
    # ********************************* lzz append1 *********************************
    for path, img, im0s, vid_cap in dataset:
        # ********************************* lzz append1 *********************************
        if PreImage is None:
            print("当前图像为第1帧 无法差帧 跳过检测！")
            PreImage = img.reshape(img.shape[1],img.shape[2],img.shape[0])
            continue
        if CurImage is None:
            print("当前图像为第2帧 无法差帧 跳过检测！")
            CurImage = img.reshape(img.shape[1],img.shape[2],img.shape[0])
            CurImageim0s = im0s
            continue
        NextImage = img.reshape(img.shape[1],img.shape[2],img.shape[0])
        NextImageim0s =im0s
        RealImage = CurImageim0s
        composed_img = compositeDiff(PreImage, CurImage, NextImage)
        cv2.imwrite(str(save_dir /(str(nowcount)+".jpg")), composed_img)
        nowcount += 1
        composed_img = composed_img.reshape(composed_img.shape[2],composed_img.shape[0],composed_img.shape[1])
        PreImage = CurImage
        CurImage = NextImage
        CurImageim0s = NextImageim0s
        # ********************************* lzz append1 *********************************
        # img = torch.from_numpy(img).to(device)
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)
        composed_img = torch.from_numpy(composed_img).to(device)
        composed_img = composed_img.half() if half else composed_img.float()  # uint8 to fp16/32
        composed_img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if composed_img.ndimension() == 3:
            composed_img = composed_img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = model(composed_img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, composed_img, RealImage)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = Path(path[i]), '%g: ' % i, RealImage[i].copy()
            else:
                p, s, im0 = Path(path), '', RealImage
            # ********************************* lzz append2 *********************************
            ImgList.append(composed_img)
            # ********************************* lzz append2 *********************************
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % composed_img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(composed_img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                # ********************************* lzz append2 *********************************
                CycleChoose = ""
                CycleConf = 0.0
                StdChoose = ""
                StdConf = 0.0
                # ********************************* lzz append2 *********************************
                for *xyxy, conf, cls in reversed(det):
                    # ********************************* lzz append2 *********************************
                    if names[int(cls)] in CycleClass and conf > CycleConf:
                        CycleChoose = names[int(cls)]
                        CycleConf = conf
                    elif names[int(cls)] in StdClass and conf > StdConf:
                        StdChoose = names[int(cls)]
                        StdConf = conf
                    # ********************************* lzz append2 *********************************
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                # ********************************* lzz append2 *********************************
                if CycleConf.cpu().numpy() >= 0.9:
                    LabelCycleList.append(CycleChoose)
                    ResLabelCycleList.append(CycleClass.index(CycleChoose))
                else:
                    LabelCycleList.append("Inter")
                    ResLabelCycleList.append(2)
                if StdChoose == "":
                    LabelStdList.append("Nstd")
                else:
                    LabelStdList.append(StdChoose)
                # ********************************* lzz append2 *********************************
            # ********************************* lzz append2 *********************************
            else:
                print("没有进来！！！！")
                LabelCycleList.append("Inter")
                LabelStdList.append("Nstd")
            # ********************************* lzz append2 *********************************
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

                    # vid_writer.write(im0)
                    vid_writer.write(im0)
    print("LabelCycleList:")
    print(len(LabelCycleList))
    print(LabelCycleList)
    print("LabelStdList:")
    print(len(LabelStdList))
    print(LabelStdList)
    print("ResLabelCycleList:")
    print(len(ResLabelCycleList))
    print(ResLabelCycleList)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
