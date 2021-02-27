import os
import json
import copy
import cv2
import numpy as np
orgjson = {
    "annotations":{}
}
Structure = {}
SectionCount = {}
def list_dir(file_dir):
    dir_list = os.listdir(file_dir)
    print("############################################################################################")
    print(file_dir)
    print("############################################################################################")
    for cur_file in dir_list:
        path = os.path.join(file_dir, cur_file)
        if os.path.isfile(path) and cur_file == "annotations.json":
            f = open(path,encoding='utf-8')
            frame = json.load(f)
            annotations = frame["annotations"]
            # 遍历所有图片 找出heart的包围盒
            for x in annotations:
                picpath = os.path.join(file_dir, x)
                if not os.path.exists(picpath):
                    print(picpath,' 不存在')
                    continue
                cur_section = annotations[x]["bodyPart"]
                if cur_section not in Structure:
                    Structure[cur_section] = {}
                    SectionCount[cur_section] = 0
                SectionCount[cur_section] += 1
                for y in annotations[x]["annotations"]:
                    if y['name'] not in Structure[cur_section]:
                        Structure[cur_section][y['name']] = 0
                    Structure[cur_section][y['name']] += 1
        if os.path.isdir(path):
            list_dir(path)

# list_dir(r'E:\hnumedical\Data\Pic_Data\All_Pic_data\已经标注\3VT切面')
# list_dir(r'E:\hnumedical\Data\Pic_Data\All_Pic_data\已经标注\四腔心水平横切面')
# list_dir(r'E:\hnumedical\Data\Pic_Data\All_Pic_data\已经标注\心底四腔心切面')
# list_dir(r'E:\hnumedical\Data\Pic_Data\All_Pic_data\已经标注\心尖四腔心切面')
# list_dir(r'E:\hnumedical\Data\Pic_Data\All_Pic_data\已经标注\胸骨旁四腔心切面')
# list_dir(r'E:\hnumedical\Data\Pic_Data\All_Pic_data\已经标注\心底短轴切面')
# list_dir(r'E:\hnumedical\Data\Pic_Data\All_Pic_data\已经标注\右室流出道切面')
# list_dir(r'E:\hnumedical\Data\Pic_Data\All_Pic_data\已经标注\左室流出道切面')
list_dir(r'E:\hnumedical\Heart_detection\web重新训练\Data\Diff_pic')
for cur_section in Structure:
    print("**********************")
    print(cur_section)
for cur_section in Structure:
    #print("**********************")
    print(cur_section,":",SectionCount[cur_section])
    #print("**********************")
    #for cur_structure in Structure[cur_section]:
        #print(cur_structure,":",Structure[cur_section][cur_structure])
   # print("***************************************************")