import glob
import json
import natsort # python3 get_best_accuracy.py 로 실행
import argparse

logs = natsort.natsorted(glob.glob('/home/daree/classifier/code/230601-train_set1-2-3_test_set4_epoch_150/4-class-soft_type-2/*PIXEL*/log.txt'))

for log in logs:
    model = log.split('/')[-2]
    with open(log) as f:
        lines = [i.strip() for i in f.readlines()]
        dict_collection = [json.loads(line) for line in lines]
        best_acc = 0
        for epoch in dict_collection:
            if best_acc < epoch.get('test_acc1'):
                best_acc = epoch.get('test_acc1')
        print(f"model: {model}, best_accuracy (test_acc1): {best_acc}")

