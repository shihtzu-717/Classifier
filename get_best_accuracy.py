import glob
import json
import argparse

logs = glob.glob('/home/daree/classifier/code/230601-train_set1-2-3_test_set4_epoch_150/2-class/*FIX2*/log.txt')

for log in logs:
    model = log.split('/')[-2]
    with open(log) as f:
        lines = [i.strip() for i in f.readlines()]
        dict_collection = [json.loads(line) for line in lines]
        best_acc = 0
        for epoch in dict_collection:
            if best_acc < epoch.get('test_acc1'):
                best_acc = epoch.get('test_acc1')
        print(f"model: {model}, best_accuracy : {best_acc}")

