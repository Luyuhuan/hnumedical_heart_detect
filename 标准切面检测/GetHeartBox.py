import os
import json
import copy
import cv2
import numpy as np
orgjson = {
    "annotations":{}
}
def list_dir(file_dir):
    dir_list = os.listdir(file_dir)
    print("############################################################################################")
    print(file_dir)
    print("############################################################################################")
    for cur_file in dir_list:
        path = os.path.join(file_dir, cur_file)
        if os.path.isfile(path) and cur_file == "annotations.json":
            # 将之前的json重命名
            new_json = os.path.join(file_dir, "organnotations_heart.json")
            os.rename(path, new_json)
            f = open(new_json,encoding='utf-8')
            frame = json.load(f)
            annotations = frame["annotations"]
            savejson = copy.deepcopy(orgjson)
            # 遍历所有图片 找出heart的包围盒
            for x in annotations:
                picpath = os.path.join(file_dir, x)
                if not os.path.exists(picpath):
                    print(picpath,' 不存在')
                    continue
                Xmin=[]
                Ymin=[]
                Xmax=[]
                Ymax=[]
                savejson["annotations"][x] = annotations[x]
                # 如果原先有心脏的框 先移除
                for y in savejson["annotations"][x]['annotations']:
                    if y['name'] == '心脏':
                        savejson["annotations"][x]['annotations'].remove(y)
                # 非标准的图不用求包围盒 所以直接continue
                # if annotations[x]['standard'] == '非标准':
                #     continue
                for y in savejson["annotations"][x]['annotations']:
                    # 脊柱、肋骨不用放进包围盒中
                    if y['name'] == '脊柱' or y['name'] == '肋骨':
                        continue
                    Xmin.append(float((y['start']).split(',')[0]))
                    Ymin.append(float((y['start']).split(',')[1]))
                    Xmax.append(float((y['end']).split(',')[0]))
                    Ymax.append(float((y['end']).split(',')[1]))
                # 找到小结构则求解包围盒
                if len(Xmin) > 0:
                    img = cv2.imdecode(np.fromfile(picpath, dtype=np.uint8), 1)
                    sp = img.shape
                    height = sp[0]
                    width = sp[1]
                    X1 = max((min(Xmin) - 10),0)
                    Y1 = max((min(Ymin) - 10),0)
                    X2 = min((max(Xmax) + 10),width)
                    Y2 = min((max(Ymax) + 10),height)
                    heartbox = {"type": 2, "name": "心脏", "alias": "心脏", "color": "0,1,0",
                                "start": str(X1)+','+ str(Y1), "end":str(X2)+','+str(Y2),
                                "zDepth": 0, "class": 10, "rotation": 0}
                    savejson["annotations"][x]['annotations'].append(heartbox)
            with open(file_dir + '/' +'annotations.json',"w",encoding='utf-8') as f:
                json.dump(savejson,f,ensure_ascii=False,sort_keys=True,indent=4)
            f.close()
        if os.path.isdir(path):
            list_dir(path)

list_dir(r'E:/hnumedical/Data/质检模型数据')