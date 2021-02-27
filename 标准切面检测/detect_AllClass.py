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
# ************************ lzz append1 ************************
import copy
import os
import json
annotationsorg = {
"config": {
"bodyPart": {
"心胸": {"index": 1, "annotationParts": [
{"name": "四腔心水平横切面", "showOtherInfo": 1, "subclass": [],
"config": [{"name": "左心室", "alias": "", "color": "255,95,0,25"}, {"name": "右心室", "alias": "", "color": "255,0,0,25"}, {"name": "左心房", "alias": "", "color": "31,0,255,25"}, {"name": "右心房", "alias": "", "color": "0,159,255,25"}, {"name": "室间隔", "alias": "", "color": "108,255,0,25"}, {"name": "脊柱", "alias": "", "color": "255,0,133,25"}, {"name": "降主动脉", "alias": "", "color": "12,0,255,25"}, {"name": "肋骨", "alias": "", "color": "0,255,178,25"}, {"name": "房室间隔十字交叉", "alias": "", "color": "255,0,0,25"}, {"name": "心脏", "alias": "", "color": "255,0,0,25"}]},
{"name": "左室流出道切面", "showOtherInfo": 1, "subclass": [],
"config": [{"name": "左心室", "alias": "", "color": "255,95,0,25"}, {"name": "右心室", "alias": "", "color": "255,0,0,25"}, {"name": "左室流出道及主动脉", "alias": "", "color": "255,95,0,25"}, {"name": "室间隔", "alias": "", "color": "108,255,0,25"}, {"name": "脊柱", "alias": "", "color": "255,0,133,25"}, {"name": "心脏", "alias": "", "color": "255,0,0,25"}, {"name": "左心房", "alias": "", "color": "31,0,255,25"}, {"name": "右心房", "alias": "", "color": "0,159,255,25"}, {"name": "左室及左室流出道", "alias": "", "color": "255,95,0,25"}, {"name": "右室流出道", "alias": "", "color": "255,0,0,25"}, {"name": "二尖瓣", "alias": "", "color": "0,255,25,25"}, {"name": "降主动脉", "alias": "", "color": "255,172,0,25"}, {"name": "升主动脉", "alias": "", "color": "108,0,255,25"}, {"name": "右室", "alias": "", "color": "223,0,255,25"}, {"name": "主动脉及主动脉瓣", "alias": "", "color": "12,0,255,25"}, {"name": "左心室壁", "alias": "", "color": "255,0,0,25"}, {"name": "右心室壁", "alias": "", "color": "255,0,0,25"}, {"name": "房间隔", "alias": "", "color": "255,0,0,25"}, {"name": "左室流出道", "alias": "", "color": "255,95,0,25"}, {"name": "左肺", "alias": "", "color": "0,82,255,25"}, {"name": "右肺", "alias": "", "color": "0,255,6,25"}, {"name": "左室", "alias": "", "color": "255,0,152,25"}, {"name": "主动脉瓣", "alias": "", "color": "12,0,255,128"}, {"name": "肋骨", "alias": "", "color": "108,0,255,25"}]},
{"name": "右室流出道切面", "showOtherInfo": 1, "subclass": [],
"config": [{"name": "右心室", "alias": "", "color": "255,0,0,25"}, {"name": "右室流出道及主肺动脉", "alias": "", "color": "31,0,255,25"}, {"name": "主肺动脉", "alias": "", "color": "31,0,255,25"}, {"name": "主动脉弓", "alias": "", "color": "255,0,0,25"}, {"name": "升主动脉", "alias": "", "color": "0,255,25,25"}, {"name": "上腔静脉", "alias": "", "color": "108,255,0,25"}, {"name": "脊柱", "alias": "", "color": "255,0,133,25"}, {"name": "心脏", "alias": "", "color": "255,0,0,25"}, {"name": "右室流出道", "alias": "", "color": "31,0,255,25"}, {"name": "右肺动脉", "alias": "", "color": "255,95,0,25"}, {"name": "动脉导管", "alias": "", "color": "255,0,0,25"}, {"name": "降主动脉", "alias": "", "color": "255,172,0,25"}, {"name": "左室", "alias": "", "color": "0,255,6,25"}, {"name": "右室", "alias": "", "color": "255,0,152,25"}, {"name": "肺动脉瓣", "alias": "", "color": "223,0,255,25"}, {"name": "右上腔", "alias": "", "color": "12,0,255,25"}, {"name": "气管", "alias": "", "color": "0,255,178,25"}, {"name": "左心室壁", "alias": "", "color": "255,0,0,25"}, {"name": "右心室壁", "alias": "", "color": "255,0,0,25"}, {"name": "左心室腔", "alias": "", "color": "255,0,0,25"}, {"name": "右心室腔", "alias": "", "color": "255,0,0,25"}, {"name": "室间隔", "alias": "", "color": "255,0,0,25"}, {"name": "主肺动脉及动脉导管", "alias": "", "color": "255,0,0,25"}, {"name": "右室及其流出道", "alias": "", "color": "31,0,255,128"}, {"name": "主肺动脉合并动脉导管", "alias": "", "color": "0,159,255,25"}, {"name": "左肺", "alias": "", "color": "108,0,255,25"}, {"name": "右肺", "alias": "", "color": "0,82,255,25"}, {"name": "右心房", "alias": "", "color": "31,0,255,25"}, {"name": "三尖瓣关闭", "alias": "", "color": "255,95,0,25"}, {"name": "三尖瓣开放", "alias": "", "color": "255,0,0,25"}, {"name": "肋骨", "alias": "", "color": "0,82,255,25"}, {"name": "右室壁", "alias": "", "color": "0,255,178,25"}, {"name": "左肺动脉", "alias": "", "color": "255,0,0,25"}]},
{"name": "心底短轴切面", "showOtherInfo": 1, "subclass": [],
"config": [{"name": "右心室", "alias": "", "color": "255,0,0,25"}, {"name": "主肺动脉及分叉", "alias": "", "color": "31,0,255,25"}, {"name": "升主动脉", "alias": "", "color": "255,0,133,25"}, {"name": "右心房", "alias": "", "color": "31,0,255,25"}, {"name": "脊柱", "alias": "", "color": "255,0,133,25"}, {"name": "心脏", "alias": "", "color": "255,0,0,25"}, {"name": "右室流出道", "alias": "", "color": "0,159,255,25"}, {"name": "主肺动脉及动脉导管", "alias": "", "color": "255,95,0,25"}, {"name": "右肺动脉", "alias": "", "color": "255,0,152,25"}, {"name": "降主动脉", "alias": "", "color": "108,0,255,25"}, {"name": "左肺", "alias": "", "color": "0,82,255,25"}, {"name": "右肺", "alias": "", "color": "0,255,6,25"}, {"name": "肺动脉瓣", "alias": "", "color": "223,0,255,25"}, {"name": "右室及其流出道", "alias": "", "color": "0,159,255,25"}, {"name": "右室壁", "alias": "", "color": "255,0,0,25"}, {"name": "房间隔", "alias": "", "color": "255,0,0,25"}, {"name": "室间隔", "alias": "", "color": "255,0,0,25"}, {"name": "主肺动脉合并动脉导管", "alias": "", "color": "255,95,0,25"}, {"name": "动脉导管", "alias": "", "color": "0,255,25,25"}, {"name": "三尖瓣关闭", "alias": "", "color": "255,95,0,25"}, {"name": "三尖瓣开放", "alias": "", "color": "255,0,0,25"}, {"name": "主肺动脉", "alias": "", "color": "0,255,25,25"}, {"name": "肋骨", "alias": "", "color": "0,82,255,25"}, {"name": "主动脉弓", "alias": "", "color": "223,0,255,25"}, {"name": "右上腔", "alias": "", "color": "0,82,255,25"}, {"name": "气管", "alias": "", "color": "85,255,255,25"}]},
{"name": "3VT切面", "showOtherInfo": 1, "subclass": ["心尖3VT切面", "胸骨旁3VT切面", "心底3VT切面"],
"config": [{"name": "主动脉弓", "alias": "", "color": "0,159,255,25"}, {"name": "主肺动脉及动脉导管", "alias": "", "color": "0,159,255,25"}, {"name": "上腔静脉", "alias": "", "color": "255,0,0,25"}, {"name": "气管", "alias": "", "color": "255,95,0,25"}, {"name": "脊柱", "alias": "", "color": "255,0,133,25"}, {"name": "心脏", "alias": "", "color": "255,0,0,25"}, {"name": "主肺动脉合并动脉导管", "alias": "", "color": "31,0,255,25"}, {"name": "胸腺", "alias": "", "color": "0,255,25,25"}, {"name": "胸骨", "alias": "", "color": "255,172,0,25"}, {"name": "降主动脉", "alias": "", "color": "108,0,255,25"}, {"name": "主肺动脉和动脉导管", "alias": "", "color": "255,255,0,128"}, {"name": "肺动脉及肺动脉导管", "alias": "", "color": "200,100,200,128"}]},
{"name": "左侧胸腔矢状切面", "showOtherInfo": 0, "subclass": [],
"config": [{"name": "膈肌", "alias": "", "color": "255,0,0,25"}, {"name": "心脏", "alias": "", "color": "255,95,0,25"}, {"name": "左肺", "alias": "", "color": "31,0,255,25"}, {"name": "胃", "alias": "", "color": "0,255,25,25"}, {"name": "肝脏", "alias": "", "color": "255,0,133,25"}, {"name": "右肾", "alias": "", "color": "255,172,0,25"}, {"name": "左肾", "alias": "", "color": "108,0,255,25"}, {"name": "右肺", "alias": "", "color": "0,159,255,25"}]},
{"name": "右侧胸腔矢状切面", "showOtherInfo": 0, "subclass": [],
"config": [{"name": "膈肌", "alias": "", "color": "255,0,0,25"}, {"name": "右肺", "alias": "", "color": "0,159,255,25"}, {"name": "肝脏", "alias": "", "color": "255,0,133,25"}, {"name": "心脏", "alias": "", "color": "255,95,0,25"}, {"name": "胃", "alias": "", "color": "0,255,25,25"}, {"name": "右肾", "alias": "", "color": "255,172,0,25"}, {"name": "左肾", "alias": "", "color": "108,0,255,25"}, {"name": "胆囊", "alias": "", "color": "255,0,0,25"}, {"name": "左肺", "alias": "", "color": "31,0,255,25"}]},
{"name": "膈肌冠状切面", "showOtherInfo": 0, "subclass": [],
"config": [{"name": "膈肌", "alias": "", "color": "255,0,0,25"}, {"name": "心脏", "alias": "", "color": "255,0,0,25"}, {"name": "两侧肺", "alias": "", "color": "255,0,0,25"}, {"name": "胃", "alias": "", "color": "255,0,0,25"}, {"name": "肝脏", "alias": "", "color": "255,0,0,25"}]},
{"name": "3VV切面", "showOtherInfo": 1, "subclass": [],
"config": [{"name": "主动脉、动脉导管与右肺动脉", "alias": "", "color": "255,0,0,25"}, {"name": "升主动脉", "alias": "", "color": "255,0,0,25"}, {"name": "上腔静脉", "alias": "", "color": "108,0,255,25"}, {"name": "脊柱", "alias": "", "color": "255,0,0,25"}, {"name": "主肺动脉", "alias": "", "color": "31,0,255,25"}, {"name": "右肺动脉", "alias": "", "color": "0,159,255,25"}, {"name": "动脉导管", "alias": "", "color": "255,95,0,25"}, {"name": "肋骨", "alias": "", "color": "0,255,25,25"}, {"name": "降主动脉", "alias": "", "color": "255,172,0,25"}]},
{"name": "心尖四腔心切面", "showOtherInfo": 1, "subclass": [],
"config": [{"name": "左心房", "alias": "", "color": "31,0,255,25"}, {"name": "右心房", "alias": "", "color": "0,159,255,25"}, {"name": "左心室", "alias": "", "color": "255,95,0,25"}, {"name": "右心室", "alias": "", "color": "255,0,0,25"}, {"name": "左肺", "alias": "", "color": "0,255,25,25"}, {"name": "右肺", "alias": "", "color": "255,0,133,25"}, {"name": "二尖瓣关闭", "alias": "", "color": "255,172,0,25"}, {"name": "二尖瓣开放", "alias": "", "color": "108,0,255,25"}, {"name": "三尖瓣关闭", "alias": "", "color": "0,82,255,25"}, {"name": "三尖瓣开放", "alias": "", "color": "0,255,6,25"}, {"name": "肺静脉角", "alias": "", "color": "255,0,152,25"}, {"name": "脊柱", "alias": "", "color": "223,0,255,25"}, {"name": "降主动脉", "alias": "", "color": "12,0,255,25"}, {"name": "肋骨", "alias": "", "color": "0,255,178,25"}, {"name": "室间隔", "alias": "", "color": "108,255,0,25"}, {"name": "卵圆孔瓣", "alias": "", "color": "184,0,255,25"}, {"name": "心脏面积", "alias": "", "color": "255,0,76,25"}, {"name": "胸腔面积", "alias": "", "color": "146,255,0,25"}, {"name": "脐静脉", "alias": "", "color": "255,0,0,25"}, {"name": "奇静脉", "alias": "", "color": "255,0,0,128"}, {"name": "左室壁", "alias": "", "color": "0,255,255,128"}, {"name": "右室壁", "alias": "", "color": "170,170,255,128"}, {"name": "房间隔", "alias": "", "color": "255,0,0,128"}, {"name": "左心室壁", "alias": "", "color": "255,0,0,25"}, {"name": "右心室壁", "alias": "", "color": "255,255,127,25"}]},
{"name": "胸骨旁四腔心切面", "showOtherInfo": 1, "subclass": [],
"config": [{"name": "左心房", "alias": "", "color": "31,0,255,25"}, {"name": "右心房", "alias": "", "color": "0,159,255,25"}, {"name": "左心室", "alias": "", "color": "255,95,0,25"}, {"name": "右心室", "alias": "", "color": "255,0,0,25"}, {"name": "左肺", "alias": "", "color": "0,255,25,25"}, {"name": "右肺", "alias": "", "color": "255,0,133,25"}, {"name": "二尖瓣关闭", "alias": "", "color": "255,172,0,25"}, {"name": "二尖瓣开放", "alias": "", "color": "108,0,255,25"}, {"name": "三尖瓣关闭", "alias": "", "color": "0,82,255,25"}, {"name": "三尖瓣开放", "alias": "", "color": "0,255,6,25"}, {"name": "肺静脉角", "alias": "", "color": "255,0,152,25"}, {"name": "脊柱", "alias": "", "color": "223,0,255,25"}, {"name": "降主动脉", "alias": "", "color": "12,0,255,25"}, {"name": "肋骨", "alias": "", "color": "0,255,178,25"}, {"name": "室间隔", "alias": "", "color": "108,255,0,25"}, {"name": "卵圆孔瓣", "alias": "", "color": "184,0,255,25"}, {"name": "心脏面积", "alias": "", "color": "255,0,76,25"}, {"name": "胸腔面积", "alias": "", "color": "146,255,0,25"}, {"name": "奇静脉", "alias": "", "color": "255,0,0,128"}, {"name": "右室壁", "alias": "", "color": "170,170,255,128"}, {"name": "左室壁", "alias": "", "color": "0,255,255,128"}, {"name": "房间隔", "alias": "", "color": "255,0,0,128"}, {"name": "左心室壁", "alias": "", "color": "255,0,0,25"}, {"name": "右心室壁", "alias": "", "color": "255,0,0,25"}]},
{"name": "心底四腔心切面", "showOtherInfo": 1, "subclass": [],
"config": [{"name": "左心房", "alias": "", "color": "31,0,255,25"}, {"name": "右心房", "alias": "", "color": "0,159,255,25"}, {"name": "左心室", "alias": "", "color": "255,95,0,25"}, {"name": "右心室", "alias": "", "color": "255,0,0,25"}, {"name": "左肺", "alias": "", "color": "0,255,25,25"}, {"name": "右肺", "alias": "", "color": "255,0,133,25"}, {"name": "二尖瓣关闭", "alias": "", "color": "255,172,0,25"}, {"name": "二尖瓣开放", "alias": "", "color": "108,0,255,25"}, {"name": "三尖瓣关闭", "alias": "", "color": "0,82,255,25"}, {"name": "三尖瓣开放", "alias": "", "color": "0,255,6,25"}, {"name": "肺静脉角", "alias": "", "color": "255,0,152,25"}, {"name": "脊柱", "alias": "", "color": "223,0,255,25"}, {"name": "降主动脉", "alias": "", "color": "12,0,255,25"}, {"name": "肋骨", "alias": "", "color": "0,255,178,25"}, {"name": "室间隔", "alias": "", "color": "108,255,0,25"}, {"name": "卵圆孔瓣", "alias": "", "color": "184,0,255,25"}, {"name": "心脏面积", "alias": "", "color": "255,0,76,25"}, {"name": "胸腔面积", "alias": "", "color": "146,255,0,25"}, {"name": "脐静脉", "alias": "", "color": "255,0,0,25"}, {"name": "奇静脉", "alias": "", "color": "255,0,0,128"}, {"name": "右室壁", "alias": "", "color": "170,170,255,128"}, {"name": "左室壁", "alias": "", "color": "0,255,255,128"}, {"name": "房间隔", "alias": "", "color": "255,0,0,128"}, {"name": "右室流出道", "alias": "", "color": "255,0,0,25"}, {"name": "气管", "alias": "", "color": "85,170,255,25"}, {"name": "左心室壁", "alias": "", "color": "255,0,0,25"}, {"name": "右心室壁", "alias": "", "color": "255,0,0,25"}]},
{"name": "心尖五腔心", "showOtherInfo": 1, "subclass": [],
"config": [{"name": "左心房", "alias": "", "color": "31,0,255,25"}, {"name": "右心房", "alias": "", "color": "0,159,255,25"}, {"name": "左室流出道", "alias": "", "color": "255,95,0,25"}, {"name": "右室流出道", "alias": "", "color": "255,0,0,25"}, {"name": "二尖瓣", "alias": "", "color": "0,255,25,25"}, {"name": "降主动脉", "alias": "", "color": "255,172,0,25"}, {"name": "升主动脉", "alias": "", "color": "108,0,255,25"}, {"name": "二尖瓣关闭", "alias": "", "color": "0,82,255,25"}, {"name": "左室及其流出道", "alias": "", "color": "255,95,0,128"}, {"name": "右心室", "alias": "", "color": "0,82,255,128"}, {"name": "脊柱", "alias": "", "color": "255,0,0,128"}, {"name": "左室壁", "alias": "", "color": "0,255,255,128"}, {"name": "右室壁", "alias": "", "color": "170,170,255,128"}, {"name": "房间隔", "alias": "", "color": "255,0,0,128"}, {"name": "室间隔", "alias": "", "color": "255,0,0,128"}, {"name": "肋骨", "alias": "", "color": "108,0,255,25"}, {"name": "三尖瓣", "alias": "", "color": "255,0,0,128"}, {"name": "主动脉瓣", "alias": "", "color": "170,85,255,128"}]},
{"name": "胸腔冠状切面", "showOtherInfo": 0, "subclass": [],
"config": [{"name": "心脏", "alias": "", "color": "255,95,0,25"}, {"name": "膈肌", "alias": "", "color": "255,0,0,25"}, {"name": "胃", "alias": "", "color": "0,255,25,25"}, {"name": "肝脏", "alias": "", "color": "255,0,133,25"}, {"name": "胆囊", "alias": "", "color": "255,172,0,25"}, {"name": "左肺", "alias": "", "color": "31,0,255,25"}, {"name": "右肺", "alias": "", "color": "0,159,255,25"}]}
]}},
"standard": ["标准", "基本标准", "非标准"],
"info": ["其他周期", "收缩末期", "舒张末期"]},
"annotations": {}
}
AllName = {'LV':'左心室','LA':'左心房','RV':'右心室','RA':'右心房',
           'IVS':'室间隔','RIB':'肋骨','DA':'降主动脉','SP':'脊柱',
           'RC':'房室间隔十字交叉','MPA':'主肺动脉及动脉导管','AOA':'主动脉弓','SVC':'上腔静脉',
           'TC':'气管','ASA':'升主动脉','LMPA':'左室流出道及主动脉','RPA':'右肺动脉',
           '3VT':'3VT','4C':'4C','RVOT':'RVOT','LVOT':'LVOT','SA':'SA'}
classes=['LV', 'LA', 'RV', 'RA', 'IVS', 'RIB', 'DA', 'SP', 'RC', 'MPA', 'AOA', 'SVC', 'TC', 'ASA', '3VT', '4C', 'RVOT', 'LMPA', 'LVOT', 'RPA', 'SA']
ClassName = {'3VT':'3VT切面', '4C':'四腔心水平横切面', 'RVOT':'右室流出道切面', 'LVOT':'左室流出道切面', 'SA':'心底短轴切面'}
#四腔心水平横切面 标准切面有两个肋骨 所以score(Rib)*10
Score_4C = {'LV':10, 'LA':10, 'RV':10, 'RA':10, 'RC':20, 'IVS':10, 'DA':10, 'SP':10, 'RIB':10}
Score_RVOT = {'RV':40, 'MPA':40, 'AOA':5, 'ASA':5, 'DA':5, 'SVC':5, 'SP':0}
Score_3VT = {'AOA':25, 'MPA':25, 'TC':25, 'SVC':25, 'DA':0, 'SP':0}
Score_LVOT = {'LV':35, 'RV':10, 'IVS':10, 'LMPA':35, 'SP':10}
Score_SA = {'RV':30, 'RA':8, 'MPA':30, 'ASA':8, 'DA':8, 'SP':8, 'RPA':8}
Score = {
    '4C':Score_4C,
    'RVOT':Score_RVOT,
    '3VT':Score_3VT,
    'LVOT':Score_LVOT,
    'SA':Score_SA
}
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
# ChooseStrcture = {'3VT': ['AOA', 'MPA', 'TC', 'SVC', 'DA', 'SP'],
#                   '4C': ['LV', 'LA', 'RV', 'RA', 'IVS', 'RIB', 'DA', 'SP', 'RC'],
#                   'RVOT': ['RV', 'MPA', 'AOA', 'ASA', 'DA', 'SVC', 'SP'],
#                   'LVOT': ['LV', 'RV', 'IVS', 'LMPA', 'SP'],
#                   'SA': ['RV', 'RA', 'MPA', 'ASA', 'DA', 'SP', 'RPA']}
# ************************ lzz append1 ************************
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

    # ************************ lzz append2 ************************
    AllSavejson = {}
    # ************************ lzz append2 ************************
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
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
            txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            NowAnnotations = {}
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                # ************************ lzz append3 ************************
                RIBChoose = {}
                StructureChoose = {}
                ClassChoose = ''
                ClassChooseConf = 0.0
                ClassChooseXYXY = ''
                # ************************ lzz append3 ************************
                for *xyxy, conf, cls in reversed(det):
                    # ************************ lzz append4 ************************
                    if names[int(cls)] not in classes:
                        continue
                    if names[int(cls)] == 'RIB':
                        if len(RIBChoose) < 2:
                            RIBChoose[len(RIBChoose)] = {"xyxy":xyxy,"conf":conf}
                            sorted(RIBChoose.items(), key=lambda x: x[1]["conf"], reverse=False)
                        elif len(RIBChoose) >= 2 and conf > RIBChoose[0]["conf"]:
                            RIBChoose[0] = {"xyxy":xyxy,"conf":conf}
                    elif names[int(cls)] in ClassName:
                        if conf > ClassChooseConf:
                            ClassChooseConf = conf
                            ClassChoose = names[int(cls)]
                            ClassChooseXYXY = xyxy
                    else:
                        if names[int(cls)] not in StructureChoose:
                            StructureChoose[names[int(cls)]] = {"xyxy":xyxy,"conf":conf}
                        elif names[int(cls)] in StructureChoose and conf > StructureChoose[names[int(cls)]]["conf"]:
                            StructureChoose[names[int(cls)]] = {"xyxy":xyxy,"conf":conf}
                    # ************************ lzz append4 ************************
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # ************************ 不再展示所有的检测结果 ************************
                    # if save_img or view_img:  # Add bbox to image
                    #     label = '%s %.2f' % (names[int(cls)], conf)
                    #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                # ************************ lzz append5 ************************
                ScoreAll = 0.0
                if ClassChoose != '' and ClassChooseConf != 0.0:
                    label = '%s %.2f' % (ClassChoose, ClassChooseConf)
                    plot_one_box(ClassChooseXYXY, im0, label=label, color=[255,255,0],line_thickness=3)
                else:
                    MaxLen = 0
                    for i in Score:
                        if len(StructureChoose.keys() & (Score[i]).keys()) > MaxLen:
                            MaxLen = len(StructureChoose.keys() & (Score[i]).keys())
                            ClassChoose = i
                if ClassChoose == '':
                    NowAnnotations = {"bodyPart": "其他", "subclass": "", "standard": "非标准", "info": "其他周期","annotations": []}
                else:
                    NowAnnotations["bodyPart"] = ClassName[ClassChoose]
                    NowAnnotations["subclass"] = ""
                    NowAnnotations["info"] = "其他周期"
                    NowAnnotations["annotations"] = []
                    if 'RIB' in Score[ClassChoose]:
                        for i in RIBChoose:
                            nowRib = {"type": 2, "name": "肋骨", "alias": "肋骨", "color": "0,1,0", "zDepth": 0, "class": 5, "rotation": 0}
                            if 'RIB' in Score[ClassChoose]:
                                nowRib ["start"] = (str((RIBChoose[i]['xyxy'][0]).cpu().numpy())+","+str((RIBChoose[i]['xyxy'][1]).cpu().numpy()))
                                nowRib ["end"] = (str((RIBChoose[i]['xyxy'][2]).cpu().numpy())+","+str((RIBChoose[i]['xyxy'][3]).cpu().numpy()))
                                label = '%s %.2f' % ('RIB', RIBChoose[i]['conf'])
                                plot_one_box(RIBChoose[i]['xyxy'], im0, label=label, color=colors[names.index('RIB')], line_thickness=3)
                                ScoreAll = ScoreAll + (Score[ClassChoose]['RIB']*RIBChoose[i]['conf'])/2
                                NowAnnotations["annotations"].append(nowRib)
                    for i in StructureChoose:
                        nowStructure = {"type": 2, "color": "0,1,0", "zDepth": 0,  "rotation": 0}
                        nowStructure["name"] = AllName[i]
                        nowStructure["alias"] = AllName[i]
                        nowStructure["class"] = classes.index(i)
                        if i in Score[ClassChoose] :
                            nowStructure["start"] = (str((StructureChoose[i]['xyxy'][0]).cpu().numpy()) + "," + str(
                                (StructureChoose[i]['xyxy'][1]).cpu().numpy()))
                            nowStructure["end"] = (str((StructureChoose[i]['xyxy'][2]).cpu().numpy()) + "," + str(
                                (StructureChoose[i]['xyxy'][3]).cpu().numpy()))
                            label = '%s %.2f' % (i, StructureChoose[i]['conf'])
                            plot_one_box(StructureChoose[i]['xyxy'], im0, label=label, color=colors[names.index(i)], line_thickness=3)
                            ScoreAll = ScoreAll + Score[ClassChoose][i]*StructureChoose[i]['conf']
                            NowAnnotations["annotations"].append(nowStructure)
                    if ScoreAll >= 70:
                        NowAnnotations["standard"] = "标准"
                    elif ScoreAll <70 and ScoreAll >= 55:
                        NowAnnotations["standard"] = "基本标准"
                    else:
                        NowAnnotations["standard"] = "非标准"
                    text = ClassChoose + " Score:" + '%.2f' % ScoreAll
                    sp = im0.shape
                    height = sp[0]
                    width = sp[1]
                    x_score, y_score = int(width * 1 / 10), int(height / 5)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    im0 = cv2.putText(im0, text, (x_score, y_score), font, 1.2, (255, 255, 255), 2)
            else:
                NowAnnotations = {"bodyPart": "其他", "subclass": "", "standard": "非标准", "info": "其他周期",
                                  "annotations": []}
            # ************************ lzz append5 ************************
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
                    # ************************ lzz append6 ************************
                    (picfilepath, picfilename) = os.path.split(save_path)
                    save_json_path = os.path.join(picfilepath,"annotations.json")
                    if save_json_path not in AllSavejson:
                        AllSavejson[save_json_path] = copy.deepcopy(annotationsorg)
                    AllSavejson[save_json_path]["annotations"][picfilename] = NowAnnotations
                    # ************************ lzz append6 ************************
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
                        # ************************ lzz append7 ************************
                        framecount = 0
                        (vidfilepath, vidfilename) = os.path.split(save_path)
                        (filename, extension) = os.path.splitext(vidfilename)
                        vid_jsonpath = os.path.join(vidfilepath,filename+".json")
                        if vid_jsonpath not in AllSavejson:
                            AllSavejson[vid_jsonpath] = copy.deepcopy(annotationsorg)
                        # ************************ lzz append7 ************************
                    vid_writer.write(im0)
                    # ************************ lzz append8 ************************
                    AllSavejson[vid_jsonpath]["annotations"][str(framecount)] = NowAnnotations
                    framecount += 1
                    # ************************ lzz append8 ************************
    # ************************ lzz append9 ************************
    for every_json in AllSavejson:
        with open(every_json, "w", encoding='utf-8') as f1:
            json.dump(AllSavejson[every_json], f1, ensure_ascii=False, sort_keys=True, indent=4)
        f1.close()
    # ************************ lzz append9 ************************
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
