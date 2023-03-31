import os 
import glob

from packaging import version

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]
    dirs = os.listdir(dpath)

    tags = summary_iterators[0].Tags()['scalars']

    for n, it in enumerate(summary_iterators):
        if it.Tags()['scalars'] != tags:
            summary_iterators.pop(n)
            dirs.pop(n)

    out = defaultdict(list)
    steps = defaultdict(list)

    for tag in tags:
        for events in [acc.Scalars(tag) for acc in summary_iterators]:
            out[tag].append([e.value for e in events])
            steps[tag].append([e.step for e in events])
            

    return out, steps, dirs


def to_csv(dpath):
    d, steps, dirs = tabulate_events(dpath)
    tags, values = zip(*d.items())

    col = [si for i, si in enumerate(dirs[0].split('_')) if i%2==0]
    col.extend(['last_val', 'ave_val', 'max_val', 'min_val']) 
    data = []
    data_s = []
    data_h = []
    
    for idx, tag in enumerate(tags):
        if idx == 5:
            n_val = values[idx]
            for ii, dir in enumerate(sorted(dirs)):
                val = [si for i, si in enumerate(dir.split('_')) if i%2==1]
                val.extend([n_val[ii][-1], sum(n_val[ii])/len(n_val[ii]), max(n_val[ii]), min(n_val[ii])])
                data.append(val)
                if val[-5] == '0.7':
                    data_s.append(val)
                else:
                    data_h.append(val)

            df = pd.DataFrame(np.array(data), columns=col)
            df.to_csv(get_file_path(dpath, tag))
            df = pd.DataFrame(np.array(data_s), columns=col)
            df.to_csv(get_file_path(dpath, 'soft'))
            df = pd.DataFrame(np.array(data_h), columns=col)
            df.to_csv(get_file_path(dpath, 'hard'))
            

def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


if __name__ == '__main__':
    path = "./log/"
    to_csv(path)
 




#  pad_FIX_padsize_50.0_box_True_shift_True_ratio_0.95
# pad_FIX_padsize_100.0_box_False_shift_False_ratio_0.7