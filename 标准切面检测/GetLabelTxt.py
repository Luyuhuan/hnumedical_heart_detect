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
# PIC = '/home/ultrasonic/hnumedical/ImageWare/am_heart/dataALL-0626-real'
# 四腔心水平横切面：
# 左心房、左心室、右心房、右心室、房室间隔十字交叉、室间隔、降主动脉、脊柱、肋骨
# 右室流出道切面:
# 右心室、右室流出道及主肺动脉、主动脉弓、升主动脉、降主动脉、上腔静脉、脊柱
# 3VT切面：
# 主动脉弓、主肺动脉及动脉导管、气管、上腔静脉、降主动脉、脊柱
# 左室流出道切面
# 左心室、右心室、室间隔、左室流出道及主动脉、脊柱
# 心底短轴切面   Short-axis
# 右心室、右心房、主肺动脉及动脉导管、升主动脉、降主动脉、脊柱、右肺动脉
'''
class_mapping_1={'左心室':'LV','左心房':'LA','右心室':'RV','右心房':'RA',
                 '室间隔':'IVS','肋骨':'RIB','降主动脉':'DA','脊柱':'SP',
                 '房室间隔十字交叉':'RC','主肺动脉及动脉导管':'MPA','主动脉弓':'AOA','上腔静脉':'SVC',
                 '气管':'TC','升主动脉':'ASA','右室流出道及主肺动脉':'MPA','左室流出道及主动脉':'LMPA',
                 '右肺动脉':'RPA',
                 '3VT切面心脏':'3VT','四腔心水平横切面心脏':'4C','右室流出道切面心脏':'RVOT','左室流出道切面心脏':'LVOT','心底短轴切面心脏':'SA'}
QieMian = {'3VT切面':'3VT切面', '右室流出道切面':'右室流出道切面', '四腔心水平横切面':'四腔心水平横切面', '左室流出道切面':'左室流出道切面',
           '心底短轴切面':'心底短轴切面', '心底四腔心切面':'四腔心水平横切面', '胸骨旁四腔心切面':'四腔心水平横切面', '心尖四腔心切面':'四腔心水平横切面'}
CountQieMian = {'3VT切面':0,'右室流出道切面':0,'四腔心水平横切面':0,'左室流出道切面':0,'心底短轴切面':0}
CountQieMianALL = {'3VT切面':0,'右室流出道切面':0,'四腔心水平横切面':0,'左室流出道切面':0,'心底短轴切面':0}
classes=['LV', 'LA', 'RV', 'RA',
         'IVS', 'RIB', 'DA', 'SP',
         'RC', 'MPA', 'AOA', 'SVC',
         'TC', 'ASA', 'RPA', 'LMPA',
         '4C', '3VT', 'RVOT', 'LVOT', 'SA']
'''
# 王腾师兄：
# '三血管气管切面':'主肺动脉及动脉导管', '主动脉弓', '脊柱', '上腔静脉', '降主动脉', '气管'  √
# '左室流出道切面': '左心室', '右心室', '室间隔', '左室流出道及主动脉', '脊柱' √
# '右室流出道切面':  '右心室', '脊柱', '主肺动脉及动脉导管', '主动脉弓', '降主动脉', '上腔静脉', '气管' √, '升主动脉'
# '心底短轴切面':  '右心房', '右心室', '主肺动脉及动脉导管', '脊柱', '升主动脉', '右肺动脉', '降主动脉'
# '四腔心切面':  '脊柱', '肋骨',  '左心室', '左心房', '右心室', '右心房', '室间隔', '房室间隔十字交叉', '降主动脉'
# '三血管切面':'主肺动脉', '动脉导管', '右肺动脉', '左肺动脉', '升主动脉', '降主动脉', '脊柱'
class_mapping_1={'左心室':'LV','左心房':'LA','右心室':'RV','右心房':'RA',
                 '室间隔':'IVS','肋骨':'RIB','降主动脉':'DA','脊柱':'SP',
                 '房室间隔十字交叉':'RC','主肺动脉及动脉导管':'MPA','主动脉弓':'AOA','上腔静脉':'SVC',
                 '气管':'TC','升主动脉':'ASA','右室流出道及主肺动脉':'MPA','左室流出道及主动脉':'LMPA',
                 '右肺动脉':'RPA','主肺动脉':'PA','动脉导管':'DUA','左肺动脉':'LPA',
                 '3VT切面心脏':'3VT','四腔心水平横切面心脏':'4C','右室流出道切面心脏':'RVOT',
                 '左室流出道切面心脏':'LVOT','心底短轴切面心脏':'SA','3VV切面心脏':'3VV',
                 '心底四腔心切面心脏':'4C','胸骨旁四腔心切面心脏':'4C','心尖四腔心切面心脏':'4C'}
QieMian = {'3VT切面':'3VT切面', '右室流出道切面':'右室流出道切面', '四腔心水平横切面':'四腔心水平横切面', '左室流出道切面':'左室流出道切面',
           '心底短轴切面':'心底短轴切面', '心底四腔心切面':'四腔心水平横切面', '胸骨旁四腔心切面':'四腔心水平横切面', '心尖四腔心切面':'四腔心水平横切面',
           '3VV切面':'3VV切面', }
CountQieMian = {'3VT切面':0,'右室流出道切面':0,'四腔心水平横切面':0,'左室流出道切面':0,'心底短轴切面':0,'3VV切面':0}
CountQieMianALL = {'3VT切面':0,'右室流出道切面':0,'四腔心水平横切面':0,'左室流出道切面':0,'心底短轴切面':0,'3VV切面':0}
classes=['LV', 'LA', 'RV', 'RA',
         'IVS', 'RIB', 'DA', 'SP',
         'RC', 'MPA', 'AOA', 'SVC',
         'TC', 'ASA', 'RPA', 'LMPA',
         'PA', 'DUA', 'LPA',
         '4C', '3VT', 'RVOT', 'LVOT', 'SA','3VV']
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
list_all_pic(r"E:\hnumedical\Data\质检模型数据")
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
            if (len(TestPic[every_section]) >= 0) :
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
rootpath = "../4C_3VT_RVOT_LVOT_SA_3VV"
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
                if y['name'] == '心脏':
                    classname.append(QieMian[AllInfo[pic]['bodyPart']] + y['name'])
                    locationstart.append(y['start'])
                    locationend.append(y['end'])
                elif y['name'] in class_mapping_1:
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