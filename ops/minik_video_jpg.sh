#!/bin/bash


#python3 video_jpg.py /MD1400/jhseon/datasets/minik/compress/train_256 /MD1400/jhseon/datasets/minik/images/train --file_list /MD1400/jhseon/datasets/minik/test.txt --dataset minik --parallel | tee tmp_train_log.txt

python3 video_jpg.py /data/project/datasets/minik/compress/train_256 /data/project/datasets/minik/images/train --file_list /data/project/datasets/minik/mini_train_split.txt --dataset minik --parallel | tee tmp_train_log.txt
python3 video_jpg.py /data/project/datasets/minik/compress/val_256 /data/project/datasets/minik/images/val --file_list /data/project/datasets/minik/mini_val_split.txt --dataset minik --parallel --mode val | tee tmp_val_log.txt 
#python3 video_jpg.py /dataset/jdhwang/Kinetics/kinetics-400_train /dataset/jhseon/datasets/minik/images/train --file_list /dataset/jhseon/datasets/minik/mini_train_split.txt --dataset minik --parallel | tee tmp_train_log.txt

##python3 video_jpg.py /dataset/jdhwang/Kinetics/kinetics-400_train /dataset/jhseon/datasets/minik/images/val --file_list /dataset/jhseon/datasets/minik/mini_val_split.txt --dataset minik --parallel | tee tmp_val_log.txt
