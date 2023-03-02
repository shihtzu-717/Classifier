# import cv2
# import pickle
# import statistics

# from pathlib import Path
# from collections import namedtuple, defaultdict

# RawData = namedtuple('RawData', 'data_set, label, image_path, idx, class_id, bbox')

# data_root = Path('/data/pothole_data/raw')
# data_lst = Path('/data/pothole_data/raw/data_0.1_0.1.lst')
# with data_lst.open('rb') as rf:
#     train_list, val_list, test_list, count_data = pickle.load(rf)
# w_list, h_list, width_list, height_list = [], [], [], []
# for data in train_list:
#     _, _, w, h = data.bbox
#     w_list.append(w)
#     h_list.append(h)

#     image_path = data_root / data.data_set / data.label / data.image_path
#     image = cv2.imread(str(image_path))
#     height, width, _ = image.shape
#     width *= w
#     height *= h
#     width_list.append(width)
#     height_list.append(height)
# print(f'w_list: {statistics.mean(w_list)} {min(w_list)} {max(w_list)}')
# print(f'h_list: {statistics.mean(h_list)} {min(h_list)} {max(h_list)}')
# print(f'width_list: {statistics.mean(width_list)} {min(width_list)} {max(width_list)}')
# print(f'height_list: {statistics.mean(height_list)} {min(height_list)} {max(height_list)}')
# print()
# # w_list: 0.04008586467324291 0.008177 0.198641
# # h_list: 0.037047598027127 0.007889 0.177083
# # width_list: 75.94716863131936 15.69984 368.3904
# # height_list: 39.60188356350185 6.01992 184.20048000000003

print()