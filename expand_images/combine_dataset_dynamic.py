import shutil
import os
import tqdm
from utils.data_analysis import DatasetStatistic
import json
import random


#########################
# 依据类别得分动态拷贝样本 
#########################
def combine_dataset(download_root, official_root, combine_root, labels_to_complement_number):
    if os.path.exists(combine_root):
        print('Removing %s' % combine_root)
        shutil.rmtree(combine_root)
        print('Making %s' % combine_root)
        os.mkdir(combine_root)
    else:
        print('Making %s' % combine_root)
        os.mkdir(combine_root)        

    download_files = os.listdir(download_root)
    tbar = tqdm.tqdm(labels_to_complement_number.items())
    for label, number in tbar:
        files_name = [file_name for file_name in download_files if label.split('/')[1] in file_name]
        selected_number = int(min(number, len(files_name)/2))
        selected_sample_files = random.sample(
            [file_name for file_name in files_name if file_name.endswith('jpg')],
            selected_number
        )
        selected_label_files = [file_name.replace('jpg', 'txt') for file_name in selected_sample_files]
        descript = '%s: sampled: %d / %d' % (label, selected_number, len(files_name) / 2)
        tbar.set_description(desc=descript)
        for file_name in selected_sample_files + selected_label_files:
            scr_file = os.path.join(download_root, file_name)
            target_file = os.path.join(combine_root, file_name)
            shutil.copy(scr_file, target_file)

    official_files = os.listdir(official_root)
    tbar = tqdm.tqdm(official_files)
    for official_file in tbar:
        scr_file = os.path.join(official_root, official_file)
        target_file = os.path.join(combine_root, official_file)
        shutil.copy(scr_file, target_file)


def calculate_complement_number(labels_scores, max_number, min_number):
    """
    按照分数计算需要补充的样本数目
    :param labels_scores: 各个类别的得分
    :param max_number: 最大补充样本数
    :param min_number: 最少补充样本上数
    :return:
    """
    max_score = sorted(labels_scores.values())[-1]
    min_score = sorted(labels_scores.values())[0]
    labels_to_complement_number = {}
    for key, value in labels_scores.items():
        # 得分越高，补充的样本数目越少
        complement_number = int((max_score - value) / (max_score - min_score) * (max_number - min_number) + min_number)
        labels_to_complement_number[key] = complement_number

    return labels_to_complement_number


if __name__ == "__main__":
    data_root = 'data/huawei_data/train_data'
    download_root = 'data/huawei_data/pesudeo_image'
    combine_root = 'data/huawei_data/combine_complement'
    label_id_json = 'data/huawei_data/label_id_name.json'
    score_file = 'checkpoints/se_resnext101_32x4d/log-2019-12-02T00-26-26/classes_acc.json'
    with open(label_id_json, 'r') as f:
        labels = json.load(f).values()
    labels = [label.split('/')[1] for label in labels]
    dataset_statistic = DatasetStatistic(data_root, label_id_json)
    # 最少补充样本数
    min_complement_number = 50
    # 最多补充样本数
    max_complement_number = 120
    with open(score_file, 'r') as f:
        labels_scores = json.load(f)
    labels_to_complement_number = calculate_complement_number(labels_scores, max_complement_number, min_complement_number)
    print(labels_to_complement_number)
    combine_dataset(download_root, data_root, combine_root, labels_to_complement_number)
