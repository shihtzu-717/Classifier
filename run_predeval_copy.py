import os
import sys
import glob
from pathlib import Path

date_list = [str(i) for i in range(20230713, 20230719, 20230720)]
for i in date_list:
    base = f"""python main.py \
            --model convnext_base \
            --lr 1e-6 \
            --use_cropimg=False \
            --auto_resume=False \
            --drop_path 0.2 \
            --layer_decay 0.8 \
            --test_val_ratio 0.0 1.0 \
            --use_softlabel True \
            --use_class 0 \
            --pred_eval True \
            --pred True \
            --path_type true --txt_type false \
            --eval_data_path ../nasdata/20230731_Seoul_Data/dataset_{i} \
            --pred_save False \
            --pred_save_with_conf False \
            --use_cropimg False \
            --conf 95.0"""
    
    models = [
            'results/230822_dataset_2/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_2_except-augmentation+seoul-pos+softlabel-0.7/checkpoint-best.pth',
            'results/230822_dataset_2/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_2_except-augmentation+seoul-pos+softlabel-0.8/checkpoint-best.pth',
            'results/230822_dataset_2/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_2_except-augmentation+seoul-pos+softlabel-0.9/checkpoint-best.pth',
            'results/230822_dataset_2/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_2_detection+seoul-pos+softlabel-0.7/checkpoint-best.pth',
            'results/230822_dataset_2/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_2_detection+seoul-pos+softlabel-0.8/checkpoint-best.pth',
            'results/230822_dataset_2/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_2_detection+seoul-pos+softlabel-0.9/checkpoint-best.pth',
            ]

    for m in models:
        if not os.path.exists(m):
            print(f"Check the models name {m}")
            sys.exit()
    
    org_output_dir_path = f"../res/230822_dataset_2_data/dataset_{i}_conf-95"
    org_graph_save_dir = f"../res/230822_dataset_2_graph/dataset_{i}_conf-95"

    for ckpt in models:
        nb_cls = 0
        name = ckpt.split('/')[-2]+ '_'
        ops = Path(ckpt).parts[-2]
        opsdict = dict(zip([str(d) for i,d in enumerate(str(ops).split('_')) if i%2==0], 
                        [str(d) for i, d in enumerate(ops.split('_')) if i%2==1]))
        output_dir_path = org_output_dir_path
        graph_save_dir = org_graph_save_dir
        if "nbclss_2" in ckpt:
            nb_cls = 2
            output_dir_path = os.path.join(output_dir_path, '2-class')
            graph_save_dir = os.path.join(graph_save_dir, '2-class')
        elif "nbclss_4" in ckpt and '--use_softlabel True' in base:
            nb_cls = 4 
            output_dir_path = os.path.join(output_dir_path, '4to2-class')
            graph_save_dir = os.path.join(graph_save_dir, '4to2-class')
        elif "nbclss_4" in ckpt and '--use_softlabel False' in base:
            nb_cls = 4 
            output_dir_path = os.path.join(output_dir_path, '4-class')
            graph_save_dir = os.path.join(graph_save_dir, '4-class')

        if not os.path.exists(output_dir_path):
            os.makedirs(Path(output_dir_path), exist_ok=True)
    
        if not os.path.exists(graph_save_dir):
            os.makedirs(Path(graph_save_dir), exist_ok=True)

        print(output_dir_path, graph_save_dir)

        os.system(f"""{base} \
                --resume {ckpt} \
                --padding {opsdict['pad']} \
                --padding_size {opsdict['padsize']} \
                --use_bbox {opsdict['box']} \
                --pred_eval_name {graph_save_dir}/{name} \
                --pred_save_path {output_dir_path}/{name} \
                --nb_classes {nb_cls} \
                --use_shift {opsdict['shift']}""")

date_list = [str(i) for i in range(20230713, 20230719, 20230720)]
for i in date_list:
    base = f"""python main.py \
            --model convnext_base \
            --lr 1e-6 \
            --use_cropimg=False \
            --auto_resume=False \
            --drop_path 0.2 \
            --layer_decay 0.8 \
            --test_val_ratio 0.0 1.0 \
            --use_softlabel True \
            --use_class 0 \
            --pred_eval True \
            --pred True \
            --path_type true --txt_type false \
            --eval_data_path ../nasdata/20230731_Seoul_Data/dataset_{i} \
            --pred_save False \
            --pred_save_with_conf False \
            --use_cropimg False \
            --conf 97.0"""
    
    models = [
            'results/230822_dataset_2/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_2_except-augmentation+seoul-pos+softlabel-0.7/checkpoint-best.pth',
            'results/230822_dataset_2/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_2_except-augmentation+seoul-pos+softlabel-0.8/checkpoint-best.pth',
            'results/230822_dataset_2/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_2_except-augmentation+seoul-pos+softlabel-0.9/checkpoint-best.pth',
            'results/230822_dataset_2/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_2_detection+seoul-pos+softlabel-0.7/checkpoint-best.pth',
            'results/230822_dataset_2/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_2_detection+seoul-pos+softlabel-0.8/checkpoint-best.pth',
            'results/230822_dataset_2/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_2_detection+seoul-pos+softlabel-0.9/checkpoint-best.pth',
            ]

    for m in models:
        if not os.path.exists(m):
            print(f"Check the models name {m}")
            sys.exit()
    
    org_output_dir_path = f"../res/230822_dataset_2_data/dataset_{i}_conf-97"
    org_graph_save_dir = f"../res/230822_dataset_2_graph/dataset_{i}_conf-97"

    for ckpt in models:
        nb_cls = 0
        name = ckpt.split('/')[-2]+ '_'
        ops = Path(ckpt).parts[-2]
        opsdict = dict(zip([str(d) for i,d in enumerate(str(ops).split('_')) if i%2==0], 
                        [str(d) for i, d in enumerate(ops.split('_')) if i%2==1]))
        output_dir_path = org_output_dir_path
        graph_save_dir = org_graph_save_dir
        if "nbclss_2" in ckpt:
            nb_cls = 2
            output_dir_path = os.path.join(output_dir_path, '2-class')
            graph_save_dir = os.path.join(graph_save_dir, '2-class')
        elif "nbclss_4" in ckpt and '--use_softlabel True' in base:
            nb_cls = 4 
            output_dir_path = os.path.join(output_dir_path, '4to2-class')
            graph_save_dir = os.path.join(graph_save_dir, '4to2-class')
        elif "nbclss_4" in ckpt and '--use_softlabel False' in base:
            nb_cls = 4 
            output_dir_path = os.path.join(output_dir_path, '4-class')
            graph_save_dir = os.path.join(graph_save_dir, '4-class')

        if not os.path.exists(output_dir_path):
            os.makedirs(Path(output_dir_path), exist_ok=True)
    
        if not os.path.exists(graph_save_dir):
            os.makedirs(Path(graph_save_dir), exist_ok=True)

        print(output_dir_path, graph_save_dir)

        os.system(f"""{base} \
                --resume {ckpt} \
                --padding {opsdict['pad']} \
                --padding_size {opsdict['padsize']} \
                --use_bbox {opsdict['box']} \
                --pred_eval_name {graph_save_dir}/{name} \
                --pred_save_path {output_dir_path}/{name} \
                --nb_classes {nb_cls} \
                --use_shift {opsdict['shift']}""")


    