import os

## model setting ##
# padding = ['FIX', 'PIXEL', 'FIX2']
# padding_size = [0, 50 ,100, 150, 200, 224, 256, 336, 384, 448]
# use_bbox = ['False', 'True']
# use_shift = ['False', 'True']
# soft_label_ratio = [0.9, 0.8, 0.7, 0.6]
# target_label_ratio = [1, 0.98, 0.96, 0.94, 0.92, 0.90 ]
# warmup = [5]

padding = ['PIXEL']
padding_size = [100]
use_bbox = ['True', 'False']
use_shift = ['True', 'False']
target_label_ratio = [1]
soft_label_ratio = [1]
warmup = [0]

base = """CUDA_VISIBLE_DEVICES=1 python main.py \
            --model convnext_base --drop_path 0.2 --input_size 224 \
            --batch_size 128 --lr 5e-5 --update_freq 2 \
            --epochs 30 --weight_decay 1e-8 \
            --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
            --finetune checkpoint/convnext_base_22k_224.pth \
            --data_path /home/daree/nas/ambclass/2nd_data \
            --eval_data_path /home/daree/nas/ambclass/2nd_data \
            --model_ema true --model_ema_eval true \
            --data_set image_folder \
            --nb_classes 4 \
            --use_softlabel=False \
            --use_cropimg=False \
            --auto_resume=False \
            --test_val_ratio 0.0 0.2 \
            --split_file_write=False \
            --use_cropimg False \
            --save_ckpt True \
            --lossfn BCE \
            --use_class 0 1"""

for pad in padding:
    for pad_size in padding_size:
        for bbox in use_bbox:
            for shift in use_shift:
                for target_ratio in target_label_ratio:
                    for soft_ratio in soft_label_ratio:
                        for warm in warmup:
                            name = f'pad_{pad}_padsize_{pad_size:.1f}_box_{bbox}_shift_{shift}_sratio_{soft_ratio}_tratio_{target_ratio}_warm_{warm}'
                            if not os.path.isdir(os.getcwd() + '/log/' + name):
                                os.system(f"""{base} \
                                        --padding {pad}\
                                        --padding_size {pad_size}\
                                        --use_bbox {bbox}\
                                        --use_shift {shift}\
                                        --output_dir results/b/{name} \
                                        --soft_label_ratio {soft_ratio} \
                                        --label_ratio {target_ratio} \
                                        --warmup_epochs {warm} \
                                        --log_name {name}""")
