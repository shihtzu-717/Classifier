import os

# base = """python main.py \
#             --model convnext_base --drop_path 0.2 --input_size 224 \
#             --batch_size 256 --lr 1e-5 --update_freq 2 \
#             --epochs 200 --warmup_epochs 20 --weight_decay 1e-8 \
#             --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
#             --finetune checkpoint/convnext_base_22k_224.pth \
#             --path_type false --txt_type true \
#             --data_path '../nasdata/generate_image/230801_set1-12' \
#             --model_ema false --model_ema_eval false \
#             --data_set image_folder \
#             --auto_resume=False \
#             --test_val_ratio 0.0 0.2 \
#             --split_file_write=False \
#             --save_ckpt True \
#             --lossfn BCE \
#             --use_class 0 \
#             --use_cropimg true"""

base = """CUDA_VISIBLE_DEVICES=1 python main.py \
            --model convnext_base --drop_path 0.2 --input_size 224 \
            --finetune checkpoint/convnext_base_22k_224.pth \
            --batch_size 128 --lr 5e-5 --update_freq 2 \
            --epochs 200 --warmup_epochs 20 --weight_decay 1e-8 \
            --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
            --path_type false --txt_type true \
            --train_txt_path results/230822_dataset_2/train+GAN+detection_except_augmentation+seoul_pos.txt \
            --valid_txt_path results/230822_dataset_2/valid+GAN+seoul_pos.txt \
            --model_ema false --model_ema_eval false \
            --data_set image_folder \
            --auto_resume=False \
            --test_val_ratio 0.0 0.2 \
            --split_file_write=False \
            --save_ckpt True \
            --lossfn BCE \
            --use_class 0 \
            --use_cropimg true"""

org_output_dir_name = "230822_dataset_2"
log_dir = "log_230822_dataset_2"

name1 = f'pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_2_except-augmentation+seoul-pos+softlabel-0.7'
os.system(f"""{base} \
            --padding PIXEL \
            --padding_size 100.0 \
            --use_bbox False \
            --use_shift True \
            --output_dir results/{org_output_dir_name}/2-class/{name1} \
            --soft_label_ratio 0.7 \
            --label_ratio 1.0 \
            --nb_classes 2 \
            --log_dir {log_dir} \
            --log_name {name1} \
            --use_softlabel True \
            --soft_type 1""")


name1 = f'pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_2_except-augmentation+seoul-pos+softlabel-0.8'
os.system(f"""{base} \
            --padding PIXEL \
            --padding_size 100.0 \
            --use_bbox False \
            --use_shift True \
            --output_dir results/{org_output_dir_name}/2-class/{name1} \
            --soft_label_ratio 0.8 \
            --label_ratio 1.0 \
            --nb_classes 2 \
            --log_dir {log_dir} \
            --log_name {name1} \
            --use_softlabel True \
            --soft_type 1""")


name1 = f'pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_2_except-augmentation+seoul-pos+softlabel-0.9'
os.system(f"""{base} \
            --padding PIXEL \
            --padding_size 100.0 \
            --use_bbox False \
            --use_shift True \
            --output_dir results/{org_output_dir_name}/2-class/{name1} \
            --soft_label_ratio 0.9 \
            --label_ratio 1.0 \
            --nb_classes 2 \
            --log_dir {log_dir} \
            --log_name {name1} \
            --use_softlabel True \
            --soft_type 1""")

# name2 = f'pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_4_except-augmentation+seoul-pos-softlabel'
# os.system(f"""{base} \
#             --padding PIXEL \
#             --padding_size 100.0 \
#             --use_bbox False \
#             --use_shift True \
#             --output_dir results/{org_output_dir_name}/4-class/{name2} \
#             --soft_label_ratio 1.0 \
#             --label_ratio 1.0 \
#             --nb_classes 4 \
#             --log_dir {log_dir} \
#             --log_name {name2} \
#             --use_softlabel False \
#             --soft_type 1""")