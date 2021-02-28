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

from datasetsTest import LoadImagesTest
import os
import json
import copy
CycleClass = ["Ds","Dd","Inter"]
StdClass = [ "Nstd","Std"]
global_class_mapping = ['收缩末期', '舒张末期', '其他周期']
global_std_mapping = [ '非标准', '标准']
def FindMax_DsDd(label_list, prop_list):
    index_list_choose = []
    anno_list_choose = []
    label_now = ""
    index_list_now = []
    prop_list_now = []
    for i in range(0, len(label_list)):
        if label_now == "":
            label_now = label_list[i]
            index_list_now.append(i)
            prop_list_now.append(prop_list[i])
        elif label_now == label_list[i]:
            index_list_now.append(i)
            prop_list_now.append(prop_list[i])
        elif label_now != label_list[i]:
            Max_pro_index = prop_list_now.index(max(prop_list_now))
            min_index = int(len(index_list_now) / 2)
            choose_index = int((Max_pro_index + min_index) / 2)
            if label_now != 2 and len(index_list_now) >= 1:
                index_list_choose.append(index_list_now[choose_index])
                anno_list_choose.append(label_now)
            label_now = label_list[i]
            index_list_now = []
            prop_list_now = []
            index_list_now.append(i)
            prop_list_now.append(prop_list[i])
    return index_list_choose, anno_list_choose

def save_label_json(json_path, index_list, label_list, stdFlagList):
    lines = []
    if os.path.exists(json_path):
        f = open(json_path, encoding='utf-8')
        frame = json.load(f)
        annotations = frame["annotations"]
        for x in annotations:
            json1 = '"{}": {{"bodyPart": "四腔心", "subclass": "心尖四腔心", "standard": "{}", "info": "{}", "annotations": []}},\n'.format(
                x, annotations[x]["standard"], annotations[x]["info"]
            )
            lines.append(json1)
    for index, label, stdflg in zip(index_list, label_list, stdFlagList):
        json1 = '"{}": {{"bodyPart": "四腔心", "subclass": "心尖四腔心", "standard": "{}", "info": "{}", "annotations": []}},\n'.format(
            index, global_std_mapping[stdflg], global_class_mapping[label]
        )
        if json1 not in lines:
            lines.append(json1)
    with open(json_path, 'w', encoding='utf-8') as fs:
        fs.write('{"annotations": {\n')
        fs.writelines(lines)
        fs.write('\n}}\n')

def SaveCycleVideo( video_path, Images, start_idx, end_idx):
    if start_idx == end_idx:
        return
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 10
    width = int(Images[start_idx].shape[1])
    height = int(Images[start_idx].shape[0])
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print('Failed to write video: ' + video_path)
        return
    for i in range(start_idx, end_idx + 1):
        if i >= len(Images):
            break
        image = Images[i]
        writer.write(image)
def FindCycle(Images, IndexList, AnnoList, VideoSavePath, stdFlagList):
    save_video_list = []
    save_path = os.path.join(VideoSavePath)
    save_label_json(save_path + '-detect.json', IndexList, AnnoList,stdFlagList)
    if len(IndexList) == 0:
        print('no cycles are detected')
        return [], []
    start_idx = IndexList[0]
    nRange = 0
    i = 0
    while (i + 1) < len(IndexList):
        i += 1
        if AnnoList[i] == AnnoList[i - 1]:
            start_idx = IndexList[i]
            nRange = 0
            print('frames of {} and {} are the same: {}'.format(IndexList[i - 1], IndexList[i], global_class_mapping[AnnoList[i]]))
            continue
        interval = IndexList[i] - IndexList[i - 1]
        # 因为采样过 5 30
        if interval < 2 or interval > 30:
            # maybe error, restart
            start_idx = IndexList[i]
            nRange = 0
            print('interval between {} and {} are too short or too long: {}'.format(IndexList[i - 1],IndexList[i],interval))
            continue
        nRange += 1
        # whether stop
        if nRange == 2:
            # one cycle
            print('{}-{}'.format(start_idx, IndexList[i], end=' '))
            NSTDflag = 0
            for intervalindex in range(start_idx, IndexList[i]):
                if stdFlagList[intervalindex] == 0:
                    print("Get a NSTD video between {} and {} is {}".format(start_idx,IndexList[i],intervalindex))
                    NSTDflag = NSTDflag + 1
            if NSTDflag <= 3:
                save_file_path = VideoSavePath + '-{}-{}.avi'.format(start_idx,IndexList[i])
                save_video_list.append(save_file_path)
                SaveCycleVideo(save_file_path, Images, start_idx, IndexList[i])
            i = i - 1
            nRange = 0
            start_idx = IndexList[i]

def GetAllCycle(cycleList,cycleConfList,stdList,imageList,savePath):
    indexList,cycleInfoList = FindMax_DsDd(cycleList,cycleConfList)
    print("IndexList:",indexList)
    print("CycleInfoList:",cycleInfoList)
    FindCycle(imageList, indexList, cycleInfoList, savePath, stdList)
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
        dataset = LoadImagesTest(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    CycleList = []
    CycleConfList = []
    StdList = []
    StdConfList = []
    ImageList = []
    SaveCurPath = ""
    for path, img, im0s, vid_cap in dataset:
        if img is None or im0s is None:
            print("img:",img)
            print("im0s:",im0s)
            if isinstance(vid_writer, cv2.VideoWriter) and len(CycleList):
                print("len(CycleList):",len(CycleList))
                print("CycleList:",CycleList)
                print("CycleConfList:",CycleConfList)
                print("len(StdList):",len(StdList))
                print("StdList:",StdList)
                print("StdConfList:",StdConfList)
                # CycleList[CycleConfList < 0.8] = 2
                # StdList[StdConfList < 0.8] = 0
                GetAllCycle(CycleList, CycleConfList, StdList, ImageList,SaveCurPath)
                CycleList = []
                CycleConfList = []
                StdList = []
                StdConfList = []
                vid_writer.release()
            continue
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = Path(path), '', im0s

            save_path = str(save_dir / p.name)
            SaveCurPath = save_path[:-4]
            txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            orgimg = copy.deepcopy(im0)
            ImageList.append(orgimg)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                CycleChoose = ""
                CycleConf = 0.0
                Cyclexyxy = ""
                StdChoose = ""
                StdConf = 0.0
                Stdxyxy = ""
                for *xyxy, conf, cls in reversed(det):
                    if names[int(cls)] in CycleClass and conf > CycleConf:
                        CycleChoose = names[int(cls)]
                        CycleConf = conf
                        Cyclexyxy = xyxy
                    elif names[int(cls)] in StdClass and conf > StdConf:
                        StdChoose = names[int(cls)]
                        StdConf = conf
                        Stdxyxy = xyxy
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if save_img or view_img:  # Add bbox to image
                    #     label = '%s %.2f' % (names[int(cls)], conf)
                    #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                if CycleChoose != "":
                    label = '%s %.2f' % (CycleChoose, CycleConf)
                    plot_one_box(Cyclexyxy, im0, label=label, color=colors[names.index(CycleChoose)], line_thickness=3)
                    if CycleConf.cpu().item() > 0.8:
                        CycleList.append(CycleClass.index(CycleChoose))
                        CycleConfList.append(CycleConf.cpu().item())
                    else:
                        CycleList.append(2)
                        CycleConfList.append(0.0)
                else:
                    CycleList.append(2)
                    CycleConfList.append(0.0)
                if StdChoose != "":
                    label = '%s %.2f' % (StdChoose, StdConf)
                    plot_one_box(Stdxyxy, im0, label=label, color=colors[names.index(StdChoose)], line_thickness=3)
                    if StdConf.cpu().item() > 0.8:
                        StdList.append(StdClass.index(StdChoose))
                        StdConfList.append(StdConf.cpu().item())
                    else:
                        StdList.append(0)
                        StdConfList.append(0.0)
                else:
                    StdList.append(0)
                    StdConfList.append(0.0)
            else:
                CycleList.append(2)
                CycleConfList.append(0.0)
                StdList.append(0)
                StdConfList.append(0.0)
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
                    vid_writer.write(im0)
    if isinstance(vid_writer, cv2.VideoWriter) and len(CycleList):
        print("len(CycleList):", len(CycleList))
        print("CycleList:", CycleList)
        print("CycleConfList:", CycleConfList)
        print("len(StdList):", len(StdList))
        print("StdList:", StdList)
        print("StdConfList:", StdConfList)
        GetAllCycle(CycleList, CycleConfList, StdList, ImageList,SaveCurPath)
        CycleList = []
        CycleConfList = []
        StdList = []
        StdConfList = []
        vid_writer.release()

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
