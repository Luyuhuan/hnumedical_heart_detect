import os
#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 导入CSV安装包
import csv
import csv
f = open(r'E:\hnumedical\Data\王腾师兄数据\单帧心脏切面数据\RVOT\realannotations.csv','w',encoding='utf-8',newline="")
csv_writer = csv.writer(f)
with open(r'E:\hnumedical\Data\王腾师兄数据\单帧心脏切面数据\RVOT\annotations.csv', 'r',encoding='utf-8') as f:
    reader = csv.reader(f)
    print(type(reader))
    for row in reader:
        print(row)
        realrow = []
        realrow.append(row[0])
        realrow.append(row[1])
        realrow.append(row[2])
        realrow.append(row[3])
        realrow.append(row[4])
        realrow.append("右室流出道切面")
        realrow.append("标准")
        realrow.append(row[5])
        csv_writer.writerow(realrow)
f.close()
