struct_id=${struct_id:-0}
gpu=${gpu:-0}
init=${init:-"None"}



while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done

save_dir=../../../save_dir/Graph_NAS_ImageNet_flops600M
mkdir -p ${save_dir}


resolution=224
epochs=480

horovodrun -np 8 python ts_train_image_classification.py --dataset imagenet --num_classes 1000 \
    --dist_mode horovod --workers_per_gpu 16 \
    --input_image_size ${resolution} --epochs ${epochs} --warmup 5 \
    --optimizer sgd --bn_momentum 0.01 --wd 4e-5 --nesterov --weight_init None \
    --label_smoothing --random_erase --mixup --auto_augment \
    --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
    --arch Masternet.py:MasterNet \
    --space_root ../../../proposed \
    --space GraphNet_imagenet_600M \
    --struct_id $struct_id \
    --teacher_arch geffnet_tf_efficientnet_b3_ns \
    --teacher_pretrained \
    --teacher_input_image_size 320 \
    --teacher_feature_weight 1.0 \
    --teacher_logit_weight 1.0 \
    --ts_proj_no_relu \
    --ts_proj_no_bn \
    --use_se \
    --target_downsample_ratio 16 \
    --batch_size_per_gpu 64 --save_dir ${save_dir}/ts_effnet_b3ns_epochs${epochs} \
    --world-size 8 \
