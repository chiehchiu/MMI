model_name=task4_10cls_virus
cam_path=swin_cam3d_mult_pne_all_pooling32
model_dir=./work_dirs/task4_pne_10cls_virus
dataset=pne_ct
epoch=latest.pth


if [ "$1" == "train" ]
then

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 tools/dist_train.sh configs/$dataset/$model_name.py 8 \
    --gpus 8 --work-dir $model_dir/$model_name/

elif [ "$1" == "test" ]
then

	  CUDA_VISIBLE_DEVICES=7 tools/dist_test.sh configs/$dataset/$model_name.py \
		$model_dir/$model_name/$epoch 1 \
		--out result.json --metrics 'auc_multi_cls'
fi

