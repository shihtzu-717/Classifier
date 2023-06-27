import glob
import json
import natsort # python3 get_best_accuracy.py 로 실행
import argparse

logs = natsort.natsorted(glob.glob('results/com*/*/*/log.txt'))

for log in logs:
    model = log.split('/')[-2]
    with open(log) as f:
        lines = [i.strip() for i in f.readlines()]
        dict_collection = [json.loads(line) for line in lines]
        best_acc = 0
        best_loss = 1.0
        for epoch in dict_collection:
            if best_acc < epoch.get('test_acc1'):
                best_acc = epoch.get('test_acc1')
            if epoch.get('train_loss') < best_loss:
                best_loss = epoch.get('train_loss')

        print(f"model: {model}, best_accuracy (test_acc1): {best_acc}, best_loss: {best_loss}")

