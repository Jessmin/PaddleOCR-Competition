#准备数据集
#下载数据集并解压
import glob
import json
import os
from sklearn.model_selection import train_test_split
base_dir = '/datasets/ocr/task2'
sub_dirs = ['bank','day','month','sum','year']
out_f = open(f'{base_dir}/label.txt','w')
gt_json_path = os.path.join(base_dir,'gt.json')
with open(gt_json_path,'r') as f:
    data = json.loads(f.read().replace(',\n}','}'))
    for k in data.keys():
        for item in sub_dirs:
            if item in k:
                sub_dir = item
                img_path = f'./{sub_dir}/{k}'
                line = f'{img_path}\t{data[k]}\n'
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
