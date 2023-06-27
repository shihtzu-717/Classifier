import glob
import json
import natsort # python3 get_best_accuracy.py 로 실행
import argparse

logs = natsort.natsorted(glob.glob('results/*/*/*/log.txt'))

for log in logs:
    model = log.split('/')[-2]
    with open(log) as f:
        lines = [i.strip() for i in f.readlines()]
        dict_collection = [json.loads(line) for line in lines]
        best_acc = 0
        e = 0
        for epoch in dict_collection:
            if best_acc < epoch.get('test_acc1'):
                best_acc = epoch.get('test_acc1')
                e = epoch.get("epoch")
        print(f"model: {model}, epoch: {e}, best_accuracy (test_acc1): {best_acc}")

