import glob
import json
import natsort # python3 get_best_accuracy.py 로 실행
import argparse

logs = natsort.natsorted(glob.glob('results/set1-3/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_0.7_tratio_0.95_nbclss_2/log.txt'))

for log in logs:
    model = log.split('/')[-2]
    with open(log) as f:
        lines = [i.strip() for i in f.readlines()]
        dict_collection = [json.loads(line) for line in lines]
        best_acc = 0
        best_loss = 1.0
        best_positive = 0 
        e = 0
        for epoch in dict_collection:
            if best_acc < epoch.get('test_acc1'):
                best_acc = epoch.get('test_acc1')
                e = epoch.get('epoch')

            if epoch.get('train_loss') < best_loss:
                best_loss = epoch.get('train_loss')

        print(f"model: {model}, epoch: {e},  best_accuracy (test_acc1): {best_acc}")

