import os
import glob
from pathlib import Path

base = """CUDA_VISIBLE_DEVICES=2 python main.py \
            --lr 1e-6 \
            --use_cropimg=False \
            --auto_resume=False \
            --drop_path 0.2 \
            --layer_decay 0.8 \
            --test_val_ratio 1.0 0.0 \
            --nb_classes 2 \
            --use_class 0 1 \
            --pred_eval True \
            --pred True"""

models = (glob.glob('results/b/**/checkpoint-29.pth'))
datas = ["/home/daree/nas/Classification_Model/ambclass/2nd_data"]

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
