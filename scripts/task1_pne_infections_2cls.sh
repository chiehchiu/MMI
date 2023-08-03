model_name=task1_2cls_model
cam_path=swin_cam3d_mult_pne_all_pooling32
model_dir=./work_dirs/task1_pne_infections_2cls
dataset=pne_ct
# epoch=latest.pth
epoch=epoch_30.pth
if [ "$1" == "train" ]
then
    CUDA_VISIBLE_DEVICES=0 tools/dist_train.sh configs/$dataset/$model_name.py 1 \
    --gpus 1 --work-dir $model_dir/$model_name/
elif [ "$1" == "test" ]
then

	  CUDA_VISIBLE_DEVICES=7 tools/dist_test.sh configs/$dataset/$model_name.py \
		$model_dir/$model_name/$epoch 1 \
		--out result.json --metrics 'auc'
fi