import json
import os

delete_files_path = '/media/mxq/data/competition/HuaWei/delete_bak'
dataset_split_file = '/media/mxq/project/Projects/competition/HuaWei/AI-Competition-HuaWei/dataset_split.json'

delete_files = [f for f in os.listdir(delete_files_path) if f.endswith('jpg')]
with open(dataset_split_file, 'r') as f:
    train_list, val_list = json.load(f)

delete_files_count = []
undelete_train_list = []
for fold_index, fold_list in enumerate(train_list):
    fold_sample_list = fold_list[0]
    fold_label_list = fold_list[1]
    undelete_fold_sample = []
    undelete_fold_label = []
    for sample_index, sample in enumerate(fold_sample_list):
        if sample not in delete_files:
            # 不在被删除的文件中
            undelete_fold_sample.append(sample)
            undelete_fold_label.append(fold_label_list[sample_index])
        else:
            delete_files_count.append([sample, fold_label_list[sample_index]])
            print('[Train Fold %d] Remove: %s, Label: %d' % (fold_index, sample, fold_label_list[sample_index]))
    undelete_train_list.append([undelete_fold_sample, undelete_fold_label])

undelete_val_list = []
for fold_index, fold_list in enumerate(val_list):
    fold_sample_list = fold_list[0]
    fold_label_list = fold_list[1]
    undelete_fold_sample = []
    undelete_fold_label = []
    for sample_index, sample in enumerate(fold_sample_list):
        if sample not in delete_files:
            # 不在被删除的文件中
            undelete_fold_sample.append(sample)
            undelete_fold_label.append(fold_label_list[sample_index])
        else:
            delete_files_count.append([sample, fold_label_list[sample_index]])
            print('[Val Fold %d] Remove: %s, Label: %d' % (fold_index, sample, fold_label_list[sample_index]))
    undelete_val_list.append([undelete_fold_sample, undelete_fold_label])

with open('dataset_split_delete.json', 'w') as f:
    json.dump([undelete_train_list, undelete_val_list], f, ensure_ascii=False)
pass