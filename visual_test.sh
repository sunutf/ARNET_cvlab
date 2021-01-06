if [ "$#" -eq 2 ]
then
    DATA_DIR=$1
    MODEL_DIR=$2
else
    DATA_DIR=../../datasets/activity-net-v1.3
    MODEL_DIR=../../logs_tsm/test_suite/ar-net
fi

echo "Using data path: ${DATA_DIR} and model path: ${MODEL_DIR}"

#TODO 8. New S.O.T.A. AR-Net(resnet) using updated training logics (~76.8)
python -u main_base.py actnet RGB --visual_log ../arnet_visual_log --arch resnet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list resnet50 resnet34 resnet18 --skip_list 1 2 4 --accuracy_weight 0.9 --efficency_weight 0.1 --exp_decay --init_tau 0.000001 --policy_also_backbone --policy_input_offset 3 --uniform_loss_weight 3.0 --use_gflops_loss --batch-size 1 -j 0 --gpus 0 1 2 3 --test_from ${MODEL_DIR}/reproduce_new_1222.pth.tar --data_dir ${DATA_DIR} | tee tmp_log.txt
OUTPUT8=`cat tmp_log.txt | tail -n 3`

echo $OUTPUT8
