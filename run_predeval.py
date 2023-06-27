import os
import sys
import glob
from pathlib import Path

# base = """CUDA_VISIBLE_DEVICES=2 python main.py \
#             --lr 1e-6 \
#             --use_cropimg=False \
#             --auto_resume=False \
#             --drop_path 0.2 \
#             --layer_decay 0.8 \
#             --test_val_ratio 1.0 0.0 \
#             --nb_classes 2 \
#             --use_softlabel=True \
#             --use_class 0 \
#             --eval True """

# base = """CUDA_VISIBLE_DEVICES=2 python main.py \
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
#             --pred True"""

# base = """CUDA_VISIBLE_DEVICES=2 python main.py \
#             --lr 1e-6 \
#             --use_cropimg=False \
#             --auto_resume=False \
#             --drop_path 0.2 \
#             --layer_decay 0.8 \
#             --test_val_ratio 1.0 0.0 \
#             --nb_classes 4 \
#             --use_softlabel=True \
#             --use_class 0 \
#             --pred_eval True \
#             --pred True"""

# base = """CUDA_VISIBLE_DEVICES=2 python main.py \
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
#             --pred True"""

base = """CUDA_VISIBLE_DEVICES=2 python main.py \
            --lr 1e-6 \
            --use_cropimg=False \
            --auto_resume=False \
            --drop_path 0.2 \
            --layer_decay 0.8 \
            --test_val_ratio 1.0 0.0 \
            --nb_classes 2 \
            --use_softlabel=True \
            --use_class 0 \
            --pred_eval True \
            --pred True \
            --pred_save True \
            --pred_save_path ../output/best_loss/20230605_pothole_negative \
            --pred_save_with_conf True"""

# graph_save_dir = "image/best_loss"
# graph_save_dir = "image/best_model"
graph_save_dir = "image"
# if not os.path.exists(graph_save_dir):
#     os.mkdir(graph_save_dir)

# nb_classes가 2이면 --use_softlabel=True
# nb_classes가 4이면 --use_softlabel=False
# 4to2-class 계산은 --nb_classes=4에 --use_softlabel=True

# models = ['230601-train_set1-2-3_test_set4_epoch_150/2-class/pad_FIX2_padsize_384.0_box_False_shift_True_sratio_0.8_tratio_1_nbclss_2/checkpoint-best.pth']
# models = glob.glob('results/after*/*nbclss_4*/checkpoint-best.pth')
# models = glob.glob('results/compare_acc-loss_test/2-class/*/checkpoint-best.pth')
models = glob.glob('results/compare_acc-loss_test/2-class/*/checkpoint-train_min_loss.pth')

# datas = ["/home/daree/nasdata/ambclass_update/4th_data"]
# datas = ["../nasdata/ambclass_update/4th_data"]
# datas = ["../data/set4_none_neg"]
# datas = ["../nasdata/202301614_pothole_negative/20230612_manhole_negative"]
datas = ["../nasdata/202301614_pothole_negative/20230605_pothole_negative"]



for m in models:
    if not os.path.exists(m):
        print(f"Check the models name {m}")
        sys.exit()

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
                --pred_eval_name {graph_save_dir}/{name} \
                --use_shift {opsdict['shift']}""")
