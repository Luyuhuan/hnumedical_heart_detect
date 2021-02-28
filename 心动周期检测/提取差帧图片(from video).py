import os
import json
import shutil
import cv2
import numpy as np
# from DataAugment import ResizeAndCrop
orgimagespath = r'E:\hnumedical\Heart_detection\心动周期检测\testvideo_cycle_org'
imagespath = r'E:\hnumedical\Heart_detection\心动周期检测\testvideo_cycle'
if not os.path.exists(orgimagespath):
    os.makedirs(orgimagespath)
if not os.path.exists(imagespath):
    os.makedirs(imagespath)
import copy
config_new = {
    "annotations":{}
}
Allsection = []
Choosesection = ["四腔心"]
StdDic = {"标准": "Std", "基本标准": "Std", "非标准": "Nstd"}
CycleDic = {"舒张末期": "Dd", "其他周期": "Inter", "收缩末期": "Ds"}
def list_dirSection(file_dir):
    dir_list = os.listdir(file_dir)
    for cur_file in dir_list:
        path = file_dir + '/' + cur_file
        if os.path.isfile(path) :
            if cur_file[-4:]=='.avi' or cur_file[-4:]=='.AVI':
                pathjson = path[:-4] + '.json'
                f = open(pathjson,encoding='utf-8')
                frame = json.load(f)
                annotations = frame["annotations"]
                for i in annotations:
                    if annotations[i]["bodyPart"] not in Allsection:
                        Allsection.append(annotations[i]["bodyPart"])
        if os.path.isdir(path):
            list_dirSection(path)
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
def list_dir(file_dir):
    dir_list = os.listdir(file_dir)
    orgsave_json = copy.deepcopy(config_new)
    save_json = copy.deepcopy(config_new)
    for cur_file in dir_list:
        path = file_dir + '/' + cur_file
        if os.path.isfile(path) :
            # print(cur_file[-4:])
            if cur_file[-4:]=='.avi' or cur_file[-4:]=='.AVI':
                pathjson = path[:-4] + '.json'
                f = open(pathjson,encoding='utf-8')
                frame = json.load(f)
                annotations = frame["annotations"]
                # **********************************选择差帧数据**************************************************
                CycleCount = 0
                ChooseFrame = []
                LastIndex = -1
                three_cycle = []
                three_index = []
                for i in annotations:
                    if annotations[i]["info"] in ["收缩末期","舒张末期"]:
                        # 间隔大于50的、小于3的是过长、过短的间隔，应当舍弃
                        # LastIndex初始值设为-2 是因为第0帧容易出错 0、1帧直接舍弃
                        interval = int(i) - LastIndex
                        if interval >= 50 or interval <= 2:
                            three_cycle = []
                            continue
                        # 如果跟上一个周期一样 就舍弃 并且把之前存入的 删除
                        now_cycle = annotations[i]["info"]
                        if len(three_cycle) and now_cycle == three_cycle[-1]:
                            three_cycle = []
                            continue
                        # 筛选当前训练的切面
                        if annotations[i]["bodyPart"] not in Choosesection:
                            three_cycle = []
                            continue
                        # 如果都符合要求 先加入当前周期统计中
                        three_cycle.append(now_cycle)
                        three_index.append(int(i))
                        LastIndex = int(i)
                        # 当找到一个符合要求的完整周期 就把这个周期内的三个末期图片、两个中间周期的图片加入选择的list中
                        if len(three_cycle) == 3:
                            ChooseFrame.append(three_index[0])
                            ChooseFrame.append(three_index[1])
                            ChooseFrame.append(three_index[2])
                            inter1 = int((three_index[0]+three_index[1])/2)
                            inter2 = int((three_index[1]+three_index[2])/2)
                            ChooseFrame.append(inter1)
                            ChooseFrame.append(inter2)
                            CycleCount += 1
                        # 每个视频最多采样5个周期
                        if CycleCount >= 5 :
                            print("该视频已经采集5个周期图像了！")
                ChooseFrame.sort()
                # **************************************************************************************************
                print(path)
                capture = cv2.VideoCapture(path)
                if not capture.isOpened():
                    raise ValueError('Failed to open video: ' + path)
                ret, image = capture.read()
                image_list = []
                # print("ChooseFrame:",ChooseFrame)
                while ret:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image_list.append(image)
                    nImages = len(image_list)
                    if nImages == 1:
                        preImage = image
                    elif nImages == 2:
                        curImage = image
                    else:
                        nextImage = image
                        composed_img = compositeDiff(preImage, curImage, nextImage)
                        preImage = curImage
                        curImage = nextImage
                        if str((nImages-2)) in annotations and (nImages-2) in ChooseFrame:
                            orgpic_path = os.path.join(orgimagespath,cur_file[:-4] + "_%05d" % (nImages-2) + '.jpg')
                            cv2.imencode('.jpg', preImage)[1].tofile(orgpic_path)
                            orgsave_json["annotations"][cur_file[:-4] + "_%05d" % (nImages-2) + '.jpg'] = annotations[
                                str((nImages-2))]
                            pic_path = os.path.join(imagespath,cur_file[:-4] + "_%05d" % (nImages-2) + '.jpg')
                            cv2.imencode('.jpg', composed_img)[1].tofile(pic_path)
                            save_json["annotations"][cur_file[:-4] + "_%05d" % (nImages-2) + '.jpg'] = annotations[
                                str((nImages-2))]
                    ret, image = capture.read()
                # **************************************************************************************************
        if os.path.isdir(path):
            list_dir(path)
    new1 = os.path.join(orgimagespath,'organnotations.json')
    with open(new1, "w", encoding='utf-8') as f1:
        json.dump(orgsave_json, f1, ensure_ascii=False, sort_keys=True, indent=4)
    f1.close()
    new2 = os.path.join(imagespath,'organnotations.json')
    with open(new2, "w", encoding='utf-8') as f2:
        json.dump(save_json, f2, ensure_ascii=False, sort_keys=True, indent=4)
    f2.close()
    # **************************************************************************************************
def list_dirStrcture(file_dir):
    dir_list = os.listdir(file_dir)
    save_jsonStrcture = copy.deepcopy(config_new)
    for cur_file in dir_list:
        path = os.path.join(file_dir ,cur_file)
        if os.path.isfile(path) :
            if cur_file == "annotations.json":
                littlestrcture = os.path.join(file_dir ,"littleboxannotations.json")
                os.rename(path, littlestrcture)
                f = open(littlestrcture,encoding='utf-8')
                frame = json.load(f)
                annotations = frame["annotations"]
                # **********************************选择差帧数据**************************************************
                for i in annotations:
                    picpath = os.path.join(file_dir ,i)
                    save_jsonStrcture["annotations"][i] = {}
                    save_jsonStrcture["annotations"][i]["bodyPart"] = annotations[i]["bodyPart"]
                    save_jsonStrcture["annotations"][i]["info"] = annotations[i]["info"]
                    save_jsonStrcture["annotations"][i]["standard"] = annotations[i]["standard"]
                    save_jsonStrcture["annotations"][i]["subclass"] = annotations[i]["subclass"]
                    save_jsonStrcture["annotations"][i]["annotations"] = []
                    Xstdmin , Ystdmin, Xstdmax, Ystdmax = [], [], [], []
                    Xcyclemin , Ycyclemin, Xcyclemax, Ycyclemax = [], [], [], []
                    for j in annotations[i]["annotations"]:
                        # 脊柱、肋骨不用放进包围盒中
                        Xstdmin.append(float((j['start']).split(',')[0]))
                        Ystdmin.append(float((j['start']).split(',')[1]))
                        Xstdmax.append(float((j['end']).split(',')[0]))
                        Ystdmax.append(float((j['end']).split(',')[1]))
                        if j['name'] != '脊柱' and j['name'] != '肋骨':
                            Xcyclemin.append(float((j['start']).split(',')[0]))
                            Ycyclemin.append(float((j['start']).split(',')[1]))
                            Xcyclemax.append(float((j['end']).split(',')[0]))
                            Ycyclemax.append(float((j['end']).split(',')[1]))
                    img = cv2.imdecode(np.fromfile(picpath, dtype=np.uint8), 1)
                    sp = img.shape
                    height = sp[0]
                    width = sp[1]
                    if len(Xstdmin) > 0:
                        X1std = max((min(Xstdmin) - 10), 0)
                        Y1std = max((min(Ystdmin) - 10), 0)
                        X2std = min((max(Xstdmax) + 10), width)
                        Y2std = min((max(Ystdmax) + 10), height)
                        stdget = StdDic[save_jsonStrcture["annotations"][i]["standard"]]
                        stdbox = {"type": 2, "name": stdget, "alias": stdget, "color": "0,1,0",
                                    "start": str(X1std) + ',' + str(Y1std), "end": str(X2std) + ',' + str(Y2std),
                                    "zDepth": 0, "class": 10, "rotation": 0}
                        save_jsonStrcture["annotations"][i]["annotations"].append(stdbox)
                    if len(Xcyclemin) > 0:
                        X1cycle = max((min(Xcyclemin) - 10), 0)
                        Y1cycle = max((min(Ycyclemin) - 10), 0)
                        X2cycle = min((max(Xcyclemax) + 10), width)
                        Y2cycle = min((max(Ycyclemax) + 10), height)
                        cycleget = CycleDic[save_jsonStrcture["annotations"][i]["info"]]
                        cyclebox = {"type": 2, "name": cycleget, "alias": cycleget, "color": "0,1,0",
                                    "start": str(X1cycle) + ',' + str(Y1cycle), "end": str(X2cycle) + ',' + str(Y2cycle),
                                    "zDepth": 0, "class": 10, "rotation": 0}
                        save_jsonStrcture["annotations"][i]['annotations'].append(cyclebox)
                with open(path, "w", encoding='utf-8') as f1:
                    json.dump(save_jsonStrcture, f1, ensure_ascii=False, sort_keys=True, indent=4)
                f1.close()
        # **************************************************************************************************
        if os.path.isdir(path):
            list_dirStrcture(path)
        # **************************************************************************************************

# ******************* 统计当前数据集中的切面信息 *****************************
list_dirSection(r'E:\hnumedical\Heart_detection\心动周期检测\testvideo')
print("当前数据集中包含的切面有：")
for i in Allsection:
    print(i)
# ******************* 从当前的视频数据中提取所需要的差帧图像数据 *****************************
print("提取视频中的差帧数据...")
list_dir(r'E:\hnumedical\Heart_detection\心动周期检测\testvideo')

# ******************* 通过yolov5检测得到小结构的信息 *******************

# ******************* 根据小结构信息筛选出std框和cycle框 *******************
# list_dirStrcture(orgimagespath)
