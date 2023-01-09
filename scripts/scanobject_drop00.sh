CUDA_VISIBLE_DEVICES=1
drop=0.0
SAMPLING="fps"
CKPT="experiments/pretraining/pretrain_official/pretrain.pth" 


DIR="fps_T25_drop9"
python main.py  --config cfgs/classification/ScanObjectNN_presampled_T25.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop
sleep 5s

DIR="fps_T25R_drop0"
python main.py  --config cfgs/classification/ScanObjectNN_presampled_T25R.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop
sleep 5s

DIR="fps_T50R_drop0"
python main.py  --config cfgs/classification/ScanObjectNN_presampled_T50R.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop
sleep 5s

DIR="fps_T50RS_drop0"
python main.py  --config cfgs/classification/ScanObjectNN_presampled_T50RS.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop