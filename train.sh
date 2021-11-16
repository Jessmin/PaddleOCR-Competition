# recommended paddle.__version__ == 2.0.0
# python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml
cp run.sh /aiwin/ocr-A/submit/
#先保证可以有结果产出 导出基本模型
python tools/export_model.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_task2.yml
#prepare dataset.py
python prepare_dataset_task2.py
#Train
python tools/train.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_task2.yml
#模型导出
python tools/export_model.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_task2.yml