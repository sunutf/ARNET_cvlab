# TODO To train to get the AR-Net(ResNet) result (mAP~73.8) shown in Table-1 in our paper
# TODO A. train for adaptive model
# TODO A-1. prepare each base model (for specific resolution) lr=0.02 for 10 epochs
python main_base.py actnet RGB --arch resnet50 --num_segments 16 --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 10 --batch-size 48 --dropout 0.5 --consensus_type=avg --eval-freq=1 --npb --gpus 0 1 2 3 --exp_header actnet_res50_t16_epo10_224 --rescale_to 224 -j 36 --data_dir ../datasets/activity-net-v1.3 --log_dir ../logs_tsm

python main_base.py actnet RGB --arch resnet34 --num_segments 16 --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 10 --batch-size 48 --dropout 0.5 --consensus_type=avg --eval-freq=1 --npb --gpus 0 1 2 3 --exp_header actnet_res34_t16_epo10_168 --rescale_to 168 -j 36 --data_dir ../datasets/activity-net-v1.3 --log_dir ../logs_tsm

python main_base.py actnet RGB --arch resnet18 --num_segments 16 --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 10 --batch-size 48 --dropout 0.5 --consensus_type=avg --eval-freq=1 --npb --gpus 0 1 2 3 --exp_header actnet_res18_t16_epo10_112 --rescale_to 112 -j 36 --data_dir ../datasets/activity-net-v1.3 --log_dir ../logs_tsm

# TODO A-2. joint training for 50 epochs with lr=0.001 (replace the GGGG with the real datetime shown in your exp dir)
python main_base.py actnet RGB --arch resnet50 --num_segments 16 --lr 0.001 --epochs 50 --batch-size 48 -j 32 --npb --gpus 0 1 2 3 --exp_header jact4_t16_3m124_a.95e.05_ed5_ft10ds --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list resnet50 resnet34 resnet18 --skip_list 1 2 4 --accuracy_weight 0.95 --efficency_weight 0.05 --model_paths ../logs_tsm/GGGG_actnet_res50_t16_epo10_224/models/ckpt.best.pth.tar ../logs_tsm/GGGG_actnet_res34_t16_epo10_168/models/ckpt.best.pth.tar ../logs_tsm/GGGG_actnet_res18_t16_epo10_112/models/ckpt.best.pth.tar --exp_decay --init_tau 5 --policy_also_backbone --policy_input_offset 3 --uniform_loss_weight 3.0 --use_gflops_loss --random_seed 1007 --data_dir ../datasets/activity-net-v1.3 --log_dir ../logs_tsm

# TODO A-3 finetuning for 50 epochs with a smaller lr=0.0005 (also freeze the policy and change the init_tau to be the value at the epoch of achieving the best mAP in last logfile)
python main_base.py actnet RGB --arch resnet50 --num_segments 16 --lr 0.0005 --epochs 50 --batch-size 48 -j 32 --npb --gpus 0 1 2 3 --exp_header jact4_t16_3m124_a.95e.05_ed5_ft10ds_finetuning --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list resnet50 resnet34 resnet18 --skip_list 1 2 4 --accuracy_weight 0.95 --efficency_weight 0.05 --base_pretrained_from ../logs_tsm/GGGG_jact4_t16_3m124_a.95e.05_ed5_ft10ds/models/ckpt.best.pth.tar --freeze_policy --exp_decay --init_tau 0.6309 --policy_also_backbone --policy_input_offset 3 --uniform_loss_weight 3.0 --use_gflops_loss --random_seed 1007 --data_dir ../datasets/activity-net-v1.3 --log_dir ../logs_tsm

## TODO B. train for baseline model (this is for 100 epochs)
#python main_base.py actnet RGB --arch resnet50 --num_segments 16 --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 100 --batch-size 48 --dropout 0.5 --consensus_type=avg --eval-freq=1 --npb --gpus 0 1 2 3 --exp_header actnet_res50_t16_epo100 --rescale_to 224 -j 36 --data_dir ../../datasets/activity-net-v1.3 --log_dir ../../logs_tsm
