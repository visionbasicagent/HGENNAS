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

save_dir=../../../experiments/GraphNet/GraphNet_id_${struct_id}/

python train_image_classification.py --dataset cifar10 --num_classes 10 \
  --dist_mode single --workers_per_gpu 6 \
  --input_image_size 32 --epochs 1440 --warmup 5 \
  --optimizer sgd --bn_momentum 0.01 --wd 5e-4 --nesterov --weight_init $init \
  --label_smoothing --random_erase --mixup --auto_augment \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt \
  --batch_size_per_gpu 64 \
  --save_dir $save_dir/cifar10_1440epochs_None \
  --space_root ../../../proposed/ \
  --space GraphNet_CIFAR_IM \
  --struct_id $struct_id \
  --gpu $gpu

python train_image_classification.py --dataset cifar100 --num_classes 100 \
  --dist_mode single --workers_per_gpu 6 \
  --input_image_size 32 --epochs 1440 --warmup 5 \
  --optimizer sgd --bn_momentum 0.01 --wd 5e-4 --nesterov --weight_init $init \
  --label_smoothing --random_erase --mixup --auto_augment \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt \
  --batch_size_per_gpu 64 \
  --save_dir $save_dir/cifar100_1440epochs_None \
  --space_root ../../../proposed/ \
  --space GraphNet_CIFAR_IM \
  --struct_id $struct_id \
  --gpu $gpu