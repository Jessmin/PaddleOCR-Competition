from PIL import Image,ImageFont,ImageDraw
import numpy as np
from pylab import *
import cv2
import os
import random
import json
import time
img_size = 60

def rotate_bound(image, angle):
    # 旋转中心点，默认为图像中心点
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)  # 得到旋转矩阵，1.0表示与原图大小一致
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # 计算旋转后的图像大小（避免图像裁剪）
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # 调整旋转矩阵（避免图像裁剪）
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # 执行仿射变换、得到图像
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(0, 0, 0))
def get_bg():
    base_dir = '/home/zhaohj/Documents/dataset/jiyan-chara/src/labelme'
    files = os.listdir(base_dir)
    choice = files[random.randint(0, len(files)-1)]
    json_path = os.path.join(base_dir,choice)
    with open(json_path,'r') as f:
        data = json.load(f)
    shapes = data['shapes']
    img_path = data['imagePath']
    img_path = os.path.join(base_dir,img_path)
    img = cv2.imread(img_path)
    for shape in shapes:
        points = shape['points']
        pt1,pt2 = points
        img[int(pt1[1]):int(pt2[1]),int(pt1[0]):int(pt2[0])] = 255
    h = random.randint(0, img.shape[0]-img_size)
    w = random.randint(0, img.shape[1]-img_size)
    bg = img[h:h+img_size,w:w+img_size]
    return bg
def get_txt_img(txt):
    font = ImageFont.truetype('fonts/simfang.ttf',65)
    # print(font.get_variation_names())
    # font.set_variation_by_name('Italic')
    im = Image.new("RGB",(img_size,img_size))
    draw = ImageDraw.Draw(im)
    x,y=(0,0)
    draw.text((x,y), txt, font=font)
    img=np.array(im)
    img = rotate_bound(img,random.randint(-30, 30))
    img = cv2.resize(img, (img_size,img_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    img = cv2.dilate(img,kernel)
    img = 255-img

    return img

def merge(txt_img,bg):
    h,w,c = txt_img.shape
    gray = cv2.cvtColor(txt_img,cv2.COLOR_BGR2GRAY)
    for i in range(h):
        for j in range(w):
            if np.mean(txt_img[i,j])!=255:
                if gray[i,j]<200:
                    bg[i,j]=txt_img[i,j]
    return bg

def run(txt):
    #repeat 4 times
    for i in range(4):
        txt_img = get_txt_img(txt)
        bg = get_bg()
        result = merge(txt_img, bg)
        fpath = f'/home/zhaohj/Documents/dataset/jiyan-chara/TextRec/Single/Single_gen/{txt}'
        os.makedirs(fpath,exist_ok=True)
        cv2.imwrite(f'{fpath}/{time.time_ns()}.png', result)
import tqdm
with open('ppocr_keys_v1.txt','r') as f:
    lines = f.read().splitlines()
    for line in tqdm.tqdm(lines):
        try:
            run(line)
        except:
            pass



    


