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

# base = """CUDA_VISIBLE_DEVICES=2 python main.py \
#             --lr 1e-6 \
#             --use_cropimg=False \
#             --auto_resume=False \
#             --drop_path 0.2 \
#             --layer_decay 0.8 \
#             --test_val_ratio 0.0 0.2 \
#             --nb_classes 2 \
#             --use_softlabel True \
#             --use_class 0 \
#             --pred True \
#             --pred_eval True \
#             --eval_data_path '../nasdata/trainset/01st_data' \
#                 '../nasdata/trainset/02nd_data' \
#                 '../nasdata/trainset/03rd_data' \
#                 '../nasdata/trainset/04th_data' \
#                 '../nasdata/trainset/05th_data' \
#                 '../nasdata/trainset/05th_data' \
#                 '../nasdata/trainset/06th_data' \
#                 '../nasdata/trainset/07th_data' \
#                 '../nasdata/trainset/08th_data' \
#                 '../nasdata/jw_data/09th_data' \
#                 '../nasdata/jw_data/10th_data' \
#                 '../nasdata/jw_data/11th_data' \
#                 '../nasdata/jw_data/12th_data' \
#             --eval_not_include_neg True"""

base = """CUDA_VISIBLE_DEVICES=0 python main.py \
            --lr 1e-6 \
            --use_cropimg=False \
            --auto_resume=False \
            --drop_path 0.2 \
            --layer_decay 0.8 \
            --test_val_ratio 0.0 1.0 \
            --use_softlabel False \
            --use_class 0 \
            --pred_eval True \
            --pred True \
            --path_type true --txt_type false \
            --eval_data_path ../nasdata/generate_image/230801_set1-12 \
            --pred_save True \
            --pred_save_with_conf True \
            --use_cropimg True"""

# --pred_save True \
# --pred_save_path ../output/230601-set1-3/4to2-class-soft_type-1/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_0.9_tratio_0.95_nbclss_4_soft-type_1/230710_pothole_positive \
# --pred_save_with_conf True \    
# datas = ["../nasdata/202301614_pothole_negative/20230605_pothole_negative"]
# datas = ["../nasdata/202301614_pothole_negative/20230612_manhole_negative"]
# datas = ["../nasdata/202301614_pothole_negative/total_pothole_manhole"]
# datas = ["../nasdata/seoul_positive_testset"]
# ../nasdata/230711_total_testset
# ../nasdata/230710_pothole_positive

# graph_save_dir = "230704_set1-12/2-class/best_loss/230711_total_testset"
# graph_save_dir = "230601_set1-3/2-class/230711_total_testset"

# nb_classes가 2이면 --use_softlabel=True
# nb_classes가 4이면 --use_softlabel=False
# 4to2-class 계산은 --nb_classes=4에 --use_softlabel=True
# models = glob.glob('results/230710_set1-12/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_0.7_tratio_0.95_nbclss_2/checkpoint-best.pth')
# best_models = ['results/230710_set1-12/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_0.7_tratio_0.95_nbclss_2/checkpoint-best.pth',
#           'results/230710_set1-12/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_2/checkpoint-best.pth',
#           'results/230710_set1-12/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_2_critical/checkpoint-best.pth',
#           'results/230710_set1-12/4-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_0.9_tratio_0.95_nbclss_4/checkpoint-best.pth',
#           'results/230710_set1-12/4-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_4/checkpoint-best.pth',
#           'results/230710_set1-12/4-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_4_critical/checkpoint-best.pth',
#           ]

# loss_models = ['results/230710_set1-12/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_0.7_tratio_0.95_nbclss_2/checkpoint-valid_min_loss.pth',
#           'results/230710_set1-12/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_2/checkpoint-valid_min_loss.pth',
#           'results/230710_set1-12/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_2_critical/checkpoint-valid_min_loss.pth',
#           'results/230710_set1-12/4-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_0.9_tratio_0.95_nbclss_4/checkpoint-valid_min_loss.pth',
#           'results/230710_set1-12/4-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_4/checkpoint-valid_min_loss.pth',
#           'results/230710_set1-12/4-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_4_critical/checkpoint-valid_min_loss.pth'
#             ]
# models = ['results/230710_set1-12/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_0.7_tratio_0.95_nbclss_2/checkpoint-best.pth',
#           'results/230710_set1-12/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_0.7_tratio_0.95_nbclss_2/checkpoint-valid_min_loss.pth',
#           'results/230710_set1-12/4-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_0.9_tratio_0.95_nbclss_4/checkpoint-best.pth',
#           'results/230710_set1-12/4-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_0.9_tratio_0.95_nbclss_4/checkpoint-valid_min_loss.pth',
#           'results/230710_set1-12/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_2/checkpoint-best.pth',
#           'results/230710_set1-12/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_2/checkpoint-valid_min_loss.pth',
#           'results/230710_set1-12/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_2_critical/checkpoint-best.pth',
#           'results/230710_set1-12/2-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_2_critical/checkpoint-valid_min_loss.pth',
#           'results/230710_set1-12/4-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_4/checkpoint-best.pth',
#           'results/230710_set1-12/4-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_4/checkpoint-valid_min_loss.pth',
#           'results/230710_set1-12/4-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_4_critical/checkpoint-best.pth',
#           'results/230710_set1-12/4-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_4_critical/checkpoint-valid_min_loss.pth',
#         ]

models = [
          'results/230710_set1-12/4-class/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1_tratio_1_nbclss_4/checkpoint-valid_min_loss.pth',
        ]

for m in models:
    if not os.path.exists(m):
        print(f"Check the models name {m}")
        sys.exit()
org_output_dir_path = "../res/generate_image"
org_graph_save_dir = "../res/generate_image"

for ckpt in models:
    nb_cls = 0
    name = ckpt.split('/')[-2]+ '_'
    ops = Path(ckpt).parts[-2]
    opsdict = dict(zip([str(d) for i,d in enumerate(str(ops).split('_')) if i%2==0], 
                       [str(d) for i, d in enumerate(ops.split('_')) if i%2==1]))
    output_dir_path = org_output_dir_path
    graph_save_dir = org_graph_save_dir
    if "2-class" in ckpt:
        nb_cls = 2
        output_dir_path = os.path.join(output_dir_path, '2-class')
        graph_save_dir = os.path.join(graph_save_dir, '2-class')
    elif "4-class" in ckpt:
        nb_cls = 4 
        output_dir_path = os.path.join(output_dir_path, '4to2-class')
        graph_save_dir = os.path.join(graph_save_dir, '4to2-class')
    if "checkpoint-valid_min_loss.pth" in ckpt:
        output_dir_path = os.path.join(output_dir_path, 'loss_model')
        graph_save_dir = os.path.join(graph_save_dir, 'loss_model')
    elif "checkpoint-best.pth" in ckpt:
        output_dir_path = os.path.join(output_dir_path, 'best_model')
        graph_save_dir = os.path.join(graph_save_dir, 'best_model')

    if not os.path.exists(output_dir_path):
        os.makedirs(Path(output_dir_path), exist_ok=True)
    
    if not os.path.exists(graph_save_dir):
        os.makedirs(Path(graph_save_dir), exist_ok=True)

    os.system(f"""{base} \
            --resume {ckpt} \
            --padding {opsdict['pad']} \
            --padding_size {opsdict['padsize']} \
            --use_bbox {opsdict['box']} \
            --pred_eval_name {graph_save_dir}/{name} \
            --pred_save_path {output_dir_path}/{name} \
            --nb_classes {nb_cls} \
            --use_shift {opsdict['shift']}""")

# for data in datas:
#     for ckpt in models:
#         name = ckpt.split('/')[-2] + '_'
#         ops = Path(ckpt).parts[-2]
#         opsdict = dict(zip([str(d) for i,d in enumerate(str(ops).split('_')) if i%2==0], 
#                            [str(d) for i, d in enumerate(ops.split('_')) if i%2==1]))
#         os.system(f"""{base} \
#                 --resume {ckpt} \
#                 --eval_data_path {data} \
#                 --data_path {data} \
#                 --padding {opsdict['pad']} \
#                 --padding_size {opsdict['padsize']} \
#                 --use_bbox {opsdict['box']} \
#                 --pred_eval_name {graph_save_dir}/{name} \
#                 --use_shift {opsdict['shift']}""")
