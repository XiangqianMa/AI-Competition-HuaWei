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
        files_name = [file_name for file_name in download_files if label in file_name]
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
    combine_root = '/media/mxq/data/competition/HuaWei/combine_complement'
    label_id_json = 'data/huawei_data/label_id_name.json'
    with open(label_id_json, 'r') as f:
        labels = json.load(f).values()
    labels = [label.split('/')[1] for label in labels]
    dataset_statistic = DatasetStatistic(data_root, label_id_json)
    # 最少补充样本数
    min_complement_number = 30
    # 最多补充样本数
    max_complement_number = 90
    labels_scores = {
        '仿唐三彩': 0.95,
        '景泰蓝': 0.95,
        '葡萄花鸟纹银香囊': 0.90,
        '西安剪纸': 0.91,
        '陕历博唐妞系列': 0.95,
        '水陆庵壁塑': 0.88,
        '汉长安城遗址': 0.86,
        '玉器': 0.91,
        '阎良甜瓜': 0.93,
        '凉鱼': 0.86,
        '羊肉泡馍': 0.95,
        '搅团': 0.75,
        '浆水面': 0.78,
        '神仙粉': 0.88,
        '荞面饸饹': 0.93,
        '蜂蜜凉粽子': 0.91,
        '醪糟': 0.92,
        '金线油塔': 0.94,
    }
    labels_to_complement_number = calculate_complement_number(labels_scores, max_complement_number, min_complement_number)
    print(labels_to_complement_number)
    combine_dataset(download_root, data_root, combine_root, labels_to_complement_number)
