import os
import json
import shutil
import cv2
import numpy as np
# from DataAugment import ResizeAndCrop
orgimagespath = r'E:\hnumedical\Heart_detection\心动周期检测\A4C_cycle_pic_org'
imagespath = r'E:\hnumedical\Heart_detection\心动周期检测\A4C_cycle_pic'
if not os.path.exists(orgimagespath):
    os.makedirs(orgimagespath)
if not os.path.exists(imagespath):
    os.makedirs(imagespath)
import copy
config_new = {
    "annotations":{}
}
Allsection = []
Choosesection = ["心尖四腔心切面"]
Choosestd = ["非标准"]
StdDic = {"标准": "Std", "基本标准": "Std", "非标准": "Nstd"}
CycleDic = {"舒张末期": "Dd", "其他周期": "Inter", "收缩末期": "Ds"}
def list_dirSection(file_dir):
    dir_list = os.listdir(file_dir)
    for cur_file in dir_list:
        path = os.path.join(file_dir, cur_file)
        if os.path.isfile(path) :
            if cur_file =='annotations.json':
                f = open(path,encoding='utf-8')
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
        path = os.path.join(file_dir, cur_file)
        if os.path.isfile(path) :
            if cur_file =='annotations.json':
                pathjson = path
                f = open(pathjson,encoding='utf-8')
                frame = json.load(f)
                annotations = frame["annotations"]
                # **********************************提取差帧数据**************************************************
                for i in annotations:
                    picpath = os.path.join(file_dir,i)
                    if not os.path.exists(picpath):
                        print("该图片不存在：",picpath)
                    if annotations[i]["bodyPart"] not in Choosesection:
                        print("该图片类型不正确：",picpath,"  -  ",annotations[i]["bodyPart"])
                    if annotations[i]["standard"] not in Choosestd:
                        print("该图片不是非标准：",picpath,"  -  ",annotations[i]["info"])
                    image = cv2.imdecode(np.fromfile(picpath, dtype=np.uint8), 1)
                    orgpic_path = os.path.join(orgimagespath, i)
                    cv2.imencode('.jpg', image)[1].tofile(orgpic_path)
                    orgsave_json["annotations"][i] = annotations[i]
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    composed_img = compositeDiff(image, image, image)
                    pic_path = os.path.join(imagespath, i)
                    cv2.imencode('.jpg', composed_img)[1].tofile(pic_path)
                    save_json["annotations"][i] = annotations[i]
                # **************************************************************************************************
        if os.path.isdir(path):
            list_dir(path)
    new1 = os.path.join(orgimagespath,'annotations.json')
    with open(new1, "w", encoding='utf-8') as f1:
        json.dump(orgsave_json, f1, ensure_ascii=False, sort_keys=True, indent=4)
    f1.close()
    new2 = os.path.join(imagespath,'annotations.json')
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
                    for j in annotations[i]["annotations"]:
                        # 脊柱、肋骨不用放进包围盒中
                        Xstdmin.append(float((j['start']).split(',')[0]))
                        Ystdmin.append(float((j['start']).split(',')[1]))
                        Xstdmax.append(float((j['end']).split(',')[0]))
                        Ystdmax.append(float((j['end']).split(',')[1]))
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
                with open(path, "w", encoding='utf-8') as f1:
                    json.dump(save_jsonStrcture, f1, ensure_ascii=False, sort_keys=True, indent=4)
                f1.close()
        # **************************************************************************************************
        if os.path.isdir(path):
            list_dirStrcture(path)
        # **************************************************************************************************

# ******************* 统计当前数据集中的切面信息 *****************************
list_dirSection(r'E:\hnumedical\Data\Pic_Data\All_Pic_data\已经标注\心尖四腔心切面\非标准')
print("当前数据集中包含的切面有：")
for i in Allsection:
    print(i)
# ******************* 从当前的图片数据中提取所需要的差帧图像数据 *****************************
print("提取非标准中的差帧数据...")
list_dir(r'E:\hnumedical\Data\Pic_Data\All_Pic_data\已经标注\心尖四腔心切面\非标准')


# ******************* 根据小结构信息筛选出nstd框 *******************
list_dirStrcture(orgimagespath)
