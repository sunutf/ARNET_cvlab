# TODO To train to get the improved result for AR-Net(ResNet) (mAP~76.8)
# TODO A. train for new adaptive model
# TODO A-1. prepare each base model (for specific resolution) for 15 epochs
#python main_base.py actnet RGB --arch resnet50 --num_segments 16 --gd 20 --lr 0.001 --wd 1e-4 --lr_steps 20 40 --epochs 100 --repeat_batch 2 --batch-size 24 --dropout 0.5 --consensus_type=avg --eval-freq=1 --npb --gpus 0 1 2 3 --exp_header actnet_res50_t16_epo100_224_lr.001 --rescale_to 224 -j 36 --data_dir ../datasets/activity-net-v1.3 --log_dir ../logs_tsm 


#python main_base.py actnet RGB --arch resnet50 --num_segments 16 --gd 20 --lr 0.001 --wd 1e-4 --lr_steps 40 80 --epochs 200 --repeat_batch 2 --batch-size 24 --dropout 0.5 --consensus_type=avg --eval-freq=1 --npb --gpus 0 1 2 3 --exp_header actnet_res50_t16_epo100_192_lr.001 --rescale_to 192 -j 36 --data_dir ../datasets/activity-net-v1.3 --log_dir ../logs_tsm 

#export NCCL_P2P_DISABLE=1
#CUDA_VISIBLE_DEVICES=1 python main_base.py actnet RGB --arch resnet50 --num_segments 16 --gd 20 --lr 0.001 --wd 1e-4 --lr_steps 40 --epochs 15 --batch-size 32 --dropout 0.5 --consensus_type=avg --eval-freq=1 --npb --gpus 0 --exp_header actnet_res50_t16_epo15_224_lr.001 --rescale_to 224 -j 36 --data_dir ../dataset/activity-net-v1.3 --log_dir ../logs_tsm --workers 4 

# python main_base.py actnet RGB --arch resnet50 --num_segments 16 --gd 20 --lr 0.001 --wd 1e-4 --lr_steps 20 40 --epochs 15 --batch-size 48 --dropout 0.5 --consensus_type=avg --eval-freq=1 --npb --gpus 0 1 2 3 --exp_header actnet_res18_t16_epo15_112_lr.001 --rescale_to 112 -j 36 --data_dir ../../datasets/activity-net-v1.3 --log_dir ../logs_tsm --workers 4 --test_from ../logs_tsm/arnet-new_actnet_res50_t16_epo15_224_lr.001/models/ckpt.best.pth.tar

# TODO A-2. joint training for 100 epochs (replace the GGGG with the real datetime shown in your exp dir)
#python main_base.py actnet RGB --arch resnet50 --num_segments 16 --lr 0.001 --epochs 100 --batch-size 24 -j 32 --npb --gpus 0 1 2 3 --exp_header jact4_t16_3m124_a.95e.05_ed5_ft15ds_lr.003_gu3_ep100 --ada_depth_skip --use_distil_loss --block_rnn_list conv_2 conv_3 conv_4 conv_5 --accuracy_weight 0.95 --efficency_weight 0.05 --model_paths ../logs_tsm/g1221-040831_actnet_res50_t16_epo15_224_lr.001/models/ckpt.best.pth.tar --exp_decay --init_tau 5 --uniform_loss_weight 0.0 --use_gflops_loss --random_seed 1007 --data_dir ../datasets/activity-net-v1.3 --log_dir ../logs_tsm  

# TODO A-2. joint training for 100 epochs (replace the GGGG with the real datetime shown in your exp dir)
CUDA_VISIBLE_DEVICES=1 python main_base.py actnet RGB --arch resnet50 --num_segments 16 --lr 0.001 --epochs 60  --batch-size 24 --repeat_batch 1 --lr_steps 30 -j 32 --npb --gpus 0 --exp_header sofnet_b24_res50_168_ep60 --ada_depth_skip --rescale_to 168 --block_rnn_list base conv_2 conv_3 conv_4 conv_5 --accuracy_weight 0.9 --efficency_weight 0.1 --model_paths ../logs_tsm/g0110-114601_actnet_res50_t16_epo15_168_lr.001/models/ckpt.best.pth.tar --exp_decay --init_tau 5.0  --uniform_loss_weight 0.0 --use_gflops_loss --random_seed 1007 --data_dir ../dataset/activity-net-v1.3 --log_dir ../logs_tsm   


## TODO B. train for new baseline model (this is also for 100 epochs)
#python main_base.py actnet RGB --arch resnet50 --num_segments 16 --gd 20 --lr 0.001 --wd 1e-4 --lr_steps 100 --epochs 120 --batch-size 48 --dropout 0.5 --consensus_type=avg --eval-freq=1 --npb --gpus 0 1 2 3 --exp_header actnet_tsn_resnet50_seg16_epo120_sz224_b48_lr.001s100 --rescale_to 224 -j 36 --data_dir ../datasets/activity-net-v1.3 --log_dir ../logs_tsm

