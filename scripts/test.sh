CUDA_VISIBLE_DEVICES=7
SAMPLING="fps"
CKPT="experiments/classification/ScanObjectNN_presampled_T50RS/fps_T50RS_drop9/ckpt-best.pth" 
DIR="test_T50RS"
python main_vis.py --vote --config cfgs/classification/ScanObjectNN_presampled_T50RS.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING


CKPT="experiments/classification/ScanObjectNN_presampled_T50R/fps_T50R_drop9/ckpt-best.pth" 
DIR="test_T50R"
python main_vis.py --vote --config cfgs/classification/ScanObjectNN_presampled_T50R.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING


CKPT="experiments/classification/ScanObjectNN_presampled_T25R/fps_T25R_drop9/ckpt-best.pth" 
DIR="test_T25R"
python main_vis.py --vote --config cfgs/classification/ScanObjectNN_presampled_T25R.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING


CKPT="experiments/classification/ScanObjectNN_presampled_T25/fps_T25_drop9/ckpt-best.pth" 
DIR="test_T25"
python main_vis.py --vote --config cfgs/classification/ScanObjectNN_presampled_T25.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING