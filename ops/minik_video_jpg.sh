#!/bin/bash

# python3 video_jpg.py /dataset/jdhwang/Kinetics/kinetics-400_train /dataset/jhseon/datasets/minik/images/train --file_list /dataset/jhseon/datasets/minik/mini_train_split.txt --dataset minik --parallel | tee tmp_train_log.txt

python3 video_jpg.py /dataset/jdhwang/Kinetics/kinetics-400_train /dataset/jhseon/datasets/minik/images/val --file_list /dataset/jhseon/datasets/minik/mini_val_split.txt --dataset minik --parallel | tee tmp_val_log.txt