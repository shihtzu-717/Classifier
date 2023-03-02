import os

base = "python -m torch.distributed.launch --nproc_per_node=2 main.py --lr 1e-6 --use_cropimg=False"
epochs = 50
set1_model = "/home/unmanned/results/set_1/b/pad_p_100_bbox_cls012_yp10n1/checkpoint-best.pth"       
set1_2_model = " /home/unmanned/results/set_2/b/pad_f2_336_shift_cls012_yp24n1tp1/checkpoint-best.pth"
total_model= "/home/daree/code/results/im224_b128_wu0/checkpoint-best.pth"

set1_data = "/home/daree/data/pothole_data/set_1/base_cls012/test/yolo"
set2_data = "/home/daree/data/pothole_data/set_2/train/base_cls012/test/yolo"

# test
data = [set1_data, set2_data]

# base = "python main.py --lr 1e-6 --use_cropimg=True --auto_resume=False --drop_path 0.2 --layer_decay 0.8 --data_set image_folder"
# inputsize = [336, 336]
# for i, ckpt in enumerate([set1_model]):
#     os.system(f"""{base} \
#             --resume {ckpt} \
#             --input_size {inputsize[i]} \
#             --eval_data_path {data[1]}\
#             --data_path {data[1]}\
#             --eval True""")


base = "python main.py --lr 1e-6 --use_cropimg=False --auto_resume=False --drop_path 0.2 --layer_decay 0.8"
inputsize = [224, 336]
data = '/home/daree/data/pothole_data/raw/'
for i, ckpt in enumerate([set1_2_model]):
    os.system(f"""{base} \
            --resume {ckpt} \
            --input_size {inputsize[i]} \
            --eval_data_path {data}\
            --data_path {data}\
            --eval True""")


# for input_size in [224, 256, 320]:
#     for warmup in [0, 10, 20]:
#         batch_size = 128
#         save_nm = f'im{input_size}_b{batch_size}_wu{warmup}'
#         output_dir = f'results/{save_nm}'

#         # train
#         log_dir = f'./log/{save_nm}'
#         base = "python -m torch.distributed.launch --nproc_per_node=2 main.py --lr 1e-6 --use_cropimg=False --enable_wandb=True "
#         os.system(f"""{base} --batch_size {batch_size} \
#                   --warmup_epochs {warmup} \
#                   --epochs {epochs} \
#                   --log_dir={log_dir} \
#                   --output_dir={output_dir} \
#                   --input_size {input_size} \
#                   --wandb_run_nm {save_nm}""")
#         #test yolo train set
#         ckpt = output_dir + '/checkpoint-best.pth'
#         log_dir = f'./log/test_{save_nm}'
#         output_dir = f'results/test_{save_nm}'
#         batch_size = 128

#         base = "python main.py --lr 1e-6 --use_cropimg=False --auto_resume=False"
#         os.system(f"""{base} \
#                    --log_dir={log_dir} \
#                    --output_dir={output_dir} \
#                    --input_size {input_size} \
#                    --wandb_run_nm {save_nm} \
#                    --resume {ckpt} \
#                    --eval True""")
       



