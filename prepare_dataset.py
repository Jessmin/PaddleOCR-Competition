#准备数据集
#下载数据集并解压
import glob
import json
import os
from sklearn.model_selection import train_test_split
base_dir = '/home/zhaohj/Documents/dataset/Kaggle/AIWIN/OCR2021/2021A_T1_Task1_数据集/训练集'
sub_dirs = ['amount','date']
out_f = open(f'{base_dir}/label.txt','w')
for sub_dir in sub_dirs:
    json_path = os.path.join(base_dir,sub_dir,'gt.json')
    with open(json_path,'r') as f:
        data = json.loads(f.read().replace(',\n}','}'))
        for k in data.keys():
            line = f'./{sub_dir}/images/{k}\t{data[k]}\n'
            out_f.write(line)
out_f.close()
with open(f'{base_dir}/label.txt','r') as f:
    lines = f.readlines()
    train,val = train_test_split(lines,test_size=0.3,random_state=0)
    
    with open(f"{base_dir}/train.txt",'w') as f1:
        for item in train:
            f1.write(item)
    with open(f'{base_dir}/val.txt','w') as f2:
        for item in val:
            f2.write(item)
