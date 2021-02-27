# -*- encoding: utf-8 -*-
import os
import json
import copy
import shutil
config_new={'annotations':{}}
Savepath = "E:/hnumedical/Data/质检模型数据"
Savejson = {}
Section = {}
PicList = {}
def list_dir(file_dir):
    dir_list = os.listdir(file_dir)
    print("############################################################################################")
    print(file_dir)
    print("############################################################################################")
    for cur_dir in dir_list:
        cur_path = os.path.join(file_dir,cur_dir)
        if os.path.isfile(cur_path):
            if cur_dir == "annotations.json":
                f = open(cur_path, encoding="utf-8")
                frame = json.load(f)
                annotations = frame["annotations"]
                for pic in annotations:
                    picpath = os.path.join(file_dir,pic)
                    # 判断图片是否存在
                    if not os.path.exists(picpath):
                        print("不存在图片 ",picpath)
                        continue
                    # 判断图片是否和之前处理的重复
                    if pic in PicList:
                        print("图片存在重复 ",picpath)
                        print(PicList[pic])
                        continue
                    else:
                        PicList[pic] = picpath
                    # 判断图片的切面类型 对各个切面进行统计
                    # 根据切面类型复制到对应文件夹内 并写下json
                    cur_section = annotations[pic]["bodyPart"]
                    cur_standard = annotations[pic]["standard"]
                    cur_section_path = os.path.join(Savepath,cur_section,cur_standard)
                    new_picpath = os.path.join(cur_section_path,pic)
                    cur_class = cur_section + "_" + cur_standard
                    cur_json = os.path.join(cur_section_path,"annotations.json")
                    if cur_class not in Section:
                        Section[cur_class] = 1
                        os.makedirs(cur_section_path)
                        Savejson[cur_json] = copy.deepcopy(config_new)
                    else:
                        Section[cur_class] = Section[cur_class] + 1
                    shutil.copy(picpath,new_picpath)
                    Savejson[cur_json]["annotations"][pic] = annotations[pic]

        elif os.path.isdir(cur_path):
            list_dir(cur_path)

list_dir(r"E:\hnumedical\Data\Pic_Data\All_Pic_data\已经标注")
list_dir(r"E:\hnumedical\Data\王腾师兄数据\单帧心脏切面数据\3VV")
list_dir(r"E:\hnumedical\Data\王腾师兄数据\单帧心脏切面数据\RVOT")

#分别将每个文件夹的json写入
for everyjson in Savejson:
    with open(everyjson, "w", encoding='utf-8') as f:
        json.dump(Savejson[everyjson], f, ensure_ascii=False, sort_keys=True, indent=4)
    f.close()

#输出统计的切面信息
for everclass in Section:
    print(everclass,":",Section[everclass])