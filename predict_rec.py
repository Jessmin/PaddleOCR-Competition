import glob
import tools.infer.predict_rec as predict_rec
import os
import cv2
import tqdm
import json


class Config(object):
    pass


cfg = Config()
cfg.rec_algorithm = "CRNN"
cfg.rec_model_dir = './output/rec_aiwin/inference'
cfg.rec_image_shape = "3, 32, 320"
cfg.rec_char_type = 'ch'
cfg.rec_batch_num = 20
cfg.max_text_length = 25
cfg.rec_char_dict_path = './ppocr/utils/ppocr_keys_v1.txt'
# cfg.rec_char_dict_path = './ppocr/utils/AIWIN_keys.txt'
cfg.use_space_char = False
cfg.use_gpu = True
cfg.gpu_mem = 4000
cfg.enable_mkldnn = False
cfg.benchmark = False
cfg.use_tensorrt = False
text_rec = predict_rec.TextRecognizer(cfg)
output_dict = {}

img_path_list = glob.glob(
    '/home/zhaohj/Documents/dataset/Kaggle/AIWIN/OCR2021/2021A_T1_Task1_数据集/测试集/*/images/*')
img_path_list.sort()
for img_path in tqdm.tqdm(img_path_list):
    _, fname = os.path.split(img_path)
    image = cv2.imread(img_path)
    result = text_rec([image])
    if len([0])==0:
        output_dict[fname] = dict(result='', confidence='1')
    result = result[0]
    txt, confidence = result[0]
    output_dict[fname] = dict(result=txt, confidence=float(confidence))
with open('answer.json', 'w') as f:
    data = json.dump(output_dict, f,indent=4, ensure_ascii=False)
