#coding=utf-8
import cv2
import re
import codecs
import os
import json
import numpy as np
import shutil
import random
import copy

class_mapping_1={"Std":"Std","Nstd":"Nstd","Dd":"Dd", "Inter": "Inter","Ds":"Ds"}
QieMian = {'心尖四腔心切面':'四腔心水平横切面'}
CountQieMian = {'四腔心水平横切面':0}
CountQieMianALL = {'四腔心水平横切面':0}
classes=['Std', 'Nstd',
         'Dd', 'Inter', 'Ds']
# AllPic存储所有图片信息(绝对路径) AllInfo存储图片对应的标注信息
AllInfo = {}
AllPic = {}
def list_all_pic(file_dir):
    dir_list = os.listdir(file_dir)
    for cur_file in dir_list:
        cur_path = os.path.join(file_dir, cur_file)
        if os.path.isfile(cur_path) and cur_file == 'annotations.json':
            f = open(cur_path,encoding='utf-8')
            frame = json.load(f)
            annotations = frame["annotations"]
            for x in annotations:
                pic_path = os.path.join(file_dir, str(x))
                if not os.path.exists(pic_path):
                    continue
                if annotations[x]['bodyPart'] not in QieMian:
                    continue
                real_section = QieMian[annotations[x]['bodyPart']]
                if real_section not in AllPic:
                    AllPic[real_section] = []
                AllPic[real_section].append(pic_path)
                AllInfo[pic_path] = annotations[x]
        if os.path.isdir(cur_path):
            list_all_pic(cur_path)  # 递归子目录
list_all_pic(r"E:\hnumedical\Heart_detection\心动周期检测\A4C_cycle")
list_all_pic(r"E:\hnumedical\Heart_detection\心动周期检测\A4C_cycle_pic")
# 打乱图片的顺序
for every_section in AllPic:
    random.shuffle(AllPic[every_section])
# 在打乱的顺序列表中挑出指定数量的test、val图片
TestPic = {}
for every_section in AllPic:
    TestPic[every_section] = []
    for pic in AllPic[every_section]:
        if (AllInfo[pic]["standard"]== "标准"):
            TestPic[every_section].append(pic)
            if (len(TestPic[every_section]) >= 100) :
                break
ValPic = {}
for every_section in AllPic:
    ValPic[every_section] = []
    for pic in AllPic[every_section]:
        if pic in TestPic[every_section]:
            continue
        if (AllInfo[pic]["standard"]== "标准"):
            ValPic[every_section].append(pic)
            if (len(ValPic[every_section]) >= 100) :
                break
TrainPic = {}
for every_section in AllPic:
    TrainPic[every_section] = []
    for pic in AllPic[every_section]:
        if (pic in TestPic[every_section]) or (pic in ValPic[every_section]):
            continue
        TrainPic[every_section].append(pic)
# 分别创建对应的文件夹
rootpath = "../A4C_cycle_data"
savetrainpathlabels = os.path.join(rootpath, "train", "labels")
savetrainpathpic = os.path.join(rootpath, "train", "images")
os.makedirs(savetrainpathlabels)
os.makedirs(savetrainpathpic)
savevalpathlabels = os.path.join(rootpath, "val", "labels")
savevalpathpic = os.path.join(rootpath, "val", "images")
os.makedirs(savevalpathlabels)
os.makedirs(savevalpathpic)
savetestpathlabels = os.path.join(rootpath, "test", "labels")
savetestpathpic = os.path.join(rootpath, "test", "images")
os.makedirs(savetestpathlabels)
os.makedirs(savetestpathpic)
trainpic = 'train.txt'
trainpictxt = open(trainpic,'w', encoding="utf-8")
valpic = 'val.txt'
valpictxt = open(valpic,'w', encoding="utf-8")
testpic = 'test.txt'
testpictxt = open(testpic,'w', encoding="utf-8")
Savepic_dic = {"train":savetrainpathpic,"val":savevalpathpic,"test":savetestpathpic}
Savelabels_dic = {"train":savetrainpathlabels,"val":savevalpathlabels,"test":savetestpathlabels}
Pictxt_dic = {"train":trainpictxt,"val":valpictxt,"test":testpictxt}
Pic_dic = {"train":TrainPic,"val":ValPic,"test":TestPic}
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

for type in Pic_dic:
    for every_section in Pic_dic[type]:
        for pic in Pic_dic[type][every_section]:
            if not os.path.exists(pic):
                print("不存在图片：",pic)
                continue
            if AllInfo[pic]['bodyPart'] not in QieMian:
                print("图片切面类型不对：",pic,AllInfo[pic]['bodyPart'])
                continue
            classname = []
            locationstart = []
            locationend = []
            flag = 0
            for y in AllInfo[pic]['annotations']:
                flag = 1
                if y['name'] in class_mapping_1:
                    classname.append(y['name'])
                    locationstart.append(y['start'])
                    locationend.append(y['end'])
            if len(classname) == 0:
                print(pic, "没有任何有效标注！！！")
                continue
            (PicPath, PicName) = os.path.split(pic)
            new_pic = os.path.join(Savepic_dic[type], PicName)
            shutil.copy(pic, new_pic)
            Pictxt_dic[type].writelines(new_pic + '\n')
            labeltxt = os.path.join(Savelabels_dic[type], PicName[:-4] + '.txt')
            writetxtlabeltxt = open(labeltxt, 'w', encoding="utf-8")
            image = cv2.imdecode(np.fromfile(new_pic, dtype=np.uint8), -1)
            sp = image.shape
            height = sp[0]
            width = sp[1]
            picsize = (width, height)
            for i in range(len(classname)):
                if classname[i] in class_mapping_1:
                    xmin = (locationstart[i].split(',')[0])
                    ymin = (locationstart[i].split(',')[1])
                    xmax = (locationend[i].split(',')[0])
                    ymax = (locationend[i].split(',')[1])
                    boxxxyy = (float(xmin), float(xmax), float(ymin), float(ymax))
                    bb = convert(picsize, boxxxyy)
                    writetxtlabeltxt.writelines(
                        str(classes.index(class_mapping_1[classname[i]])) + " " + " ".join([str(a) for a in bb]) + '\n')