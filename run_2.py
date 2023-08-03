import os

## model setting ##
# padding = ['FIX', 'PIXEL', 'FIX2']
# padding_size = [0, 50 ,100, 150, 200, 224, 256, 336, 384, 448]
# use_bbox = ['False', 'True']
# use_shift = ['False', 'True']
# soft_label_ratio = [0.9, 0.8, 0.7, 0.6]
# target_label_ratio = [1, 0.98, 0.96, 0.94, 0.92, 0.90 ]
# warmup = [5]


# padding = ['PIXEL']
# padding_size = [100]
# use_bbox = ['False']
# use_shift = ['True']
# soft_label_ratio = [0.7]
# target_label_ratio = [0.95]
# nb_classes = [4]
# soft_type = [1, 2]


# base = """CUDA_VISIBLE_DEVICES=1 python main.py \
#             --model convnext_base --drop_path 0.2 --input_size 224 \
#             --batch_size 256 --lr 5e-5 --update_freq 2 \
#             --epochs 200 --warmup_epochs 20 --weight_decay 1e-8 \
#             --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
#             --finetune checkpoint/convnext_base_22k_224.pth \\
#             --path_type true --txt_type false \
#             --data_path '/home/daree/nasdata/test/01st_data' \
#             --model_ema false --model_ema_eval false \
#             --data_set image_folder \
#             --use_cropimg=False \
#             --auto_resume=False \
#             --test_val_ratio 0.0 0.2 \
#             --split_file_write=False \
#             --use_cropimg False \
#             --save_ckpt True \
#             --lossfn BCE \
#             --use_class 0"""
base = """CUDA_VISIBLE_DEVICES=1 python main.py \
            --model convnext_base --drop_path 0.2 --input_size 224 \
            --batch_size 256 --lr 1e-5 --update_freq 2 \
            --epochs 200 --warmup_epochs 20 --weight_decay 1e-8 \
            --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
            --finetune checkpoint/convnext_base_22k_224.pth \
            --path_type false --txt_type true \
            --train_txt_path results/230710_set1-12/critical_train.txt \
            --valid_txt_path results/230710_set1-12/critical_valid.txt \
            --model_ema false --model_ema_eval false \
            --data_set image_folder \
            --use_cropimg=False \
            --auto_resume=False \
            --test_val_ratio 0.0 0.2 \
            --split_file_write=False \
            --use_cropimg False \
            --save_ckpt True \
            --lossfn BCE \
            --use_class 0"""

org_output_dir_name = "230710_set1-12"
log_dir = "log_230710_set1-12"

name1 = f'pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_2_critical_lr_1e-5'
os.system(f"""{base} \
            --padding PIXEL \
            --padding_size 100.0 \
            --use_bbox False \
            --use_shift True \
            --output_dir results/{org_output_dir_name}/2-class/{name1} \
            --soft_label_ratio 1.0 \
            --label_ratio 1.0 \
            --nb_classes 2 \
            --log_dir {log_dir} \
            --log_name {name1} \
            --use_softlabel True \
            --soft_type 1""")

name2= f'pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_4_critical_lr_1e-5'                       
os.system(f"""{base} \
            --padding PIXEL\
            --padding_size 100.0\
            --use_bbox False\
            --use_shift True\
            --output_dir results/{org_output_dir_name}/4-class/{name2} \
            --soft_label_ratio 1.0 \
            --label_ratio 1.0 \
            --nb_classes 4 \
            --log_dir {log_dir} \
            --log_name {name2} \
            --use_softlabel False \
            --soft_type 1""")

# name2 = f'pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_0.8_tratio_0.95_nbclss_4_soft-type_2'
# os.system(f"""{base} \
#             --padding PIXEL \
#             --padding_size 100.0 \
#             --use_bbox False \
#             --use_shift True \
#             --output_dir results/{org_output_dir_name}/4-class-soft_type-2/{name2} \
#             --soft_label_ratio 0.8 \
#             --label_ratio 0.95 \
#             --nb_classes 4 \
#             --log_dir {log_dir} \
#             --log_name {name2} \
#             --use_softlabel False \
#             --soft_type 2""")