import os
import glob
from pathlib import Path

# base = """CUDA_VISIBLE_DEVICES=1 python main.py \
#             --lr 1e-6 \
#             --use_cropimg=False \
#             --auto_resume=False \
#             --drop_path 0.2 \
#             --layer_decay 0.8 \
#             --test_val_ratio 1.0 0.0 \
#             --nb_classes 2 \
#             --use_softlabel=True \
#             --use_class 0 \
#             --pred_eval True \
#             --pred True \
#             --pred_save True \
#             --pred_save_path /home/daree/output/set1"""
            # --pred_save_with_conf True"""

base = """CUDA_VISIBLE_DEVICES=1 python main.py \
            --lr 1e-6 \
            --use_cropimg=False \
            --auto_resume=False \
            --drop_path 0.2 \
            --layer_decay 0.8 \
            --test_val_ratio 1.0 0.0 \
            --nb_classes 2 \
            --use_softlabel=True \
            --use_class 0 \
            --eval True """

# base = """CUDA_VISIBLE_DEVICES=1 python main.py \
#             --lr 1e-6 \
#             --use_cropimg=False \
#             --auto_resume=False \
#             --drop_path 0.2 \
#             --layer_decay 0.8 \
#             --test_val_ratio 1.0 0.0 \
#             --nb_classes 4 \
#             --use_softlabel=False \
#             --use_class 0 \
#             --pred_eval True \
#             --pred True"""


# models = (glob.glob('results/b/**/checkpoint-299.pth'))
# models = (glob.glob('results/4class_set2_2/**/checkpoint-29.pth'))
# models = glob.glob('230601-train_set1-2-3_test_set4_epoch_150/2-class/**/checkpoint-best.pth')

models = ['230601-train_set1-2-3_test_set4_epoch_150/2-class/pad_FIX2_padsize_384.0_box_False_shift_True_sratio_0.8_tratio_1_nbclss_2/checkpoint-best.pth']
datas = ["/home/daree/nasdata/ambclass_update/4th_data"]


for data in datas:
    for ckpt in models:
        name = ckpt.split('/')[-2] + '_'
        ops = Path(ckpt).parts[-2]
        opsdict = dict(zip([str(d) for i,d in enumerate(str(ops).split('_')) if i%2==0], 
                           [str(d) for i, d in enumerate(ops.split('_')) if i%2==1]))               
        os.system(f"""{base} \
                --resume {ckpt} \
                --eval_data_path {data} \
                --data_path {data} \
                --padding {opsdict['pad']} \
                --padding_size {opsdict['padsize']} \
                --use_bbox {opsdict['box']} \
                --pred_eval_name {name} \
                --use_shift {opsdict['shift']}""")
