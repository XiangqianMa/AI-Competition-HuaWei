import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
from PIL import Image
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
import random
import json


class DatasetStatistic():
    """对数据集的分布进行统计
    """
    def __init__(self, data_root, label_id_json):
        self.data_root = data_root
        self.label_id_json = label_id_json
    
    def get_download_number(self):
        """获取每一类别需要额外下载的样本数目

        Returns:
            name_dowmload_number: dir, {'大雁塔‘： 10, ...}
        """
        labels_number = self.get_label_number()
        all_number = np.asarray(list(labels_number.values()))
        max_number = np.max(all_number)
        label_to_name = self.get_label_to_name()
        name_download_number = {}
        for (label, number) in labels_number.items():
            download_number = (max_number - number) + 10
            name = label_to_name[str(label)]
            name_download_number[name.split('/')[1]] = download_number
        
        return name_download_number


    def get_label_number(self):
        """得到每一个类别对应的样本数目

        Returns:
            labels_number: dir {1: 256, 2:125, ...}
        """
        images = self.get_images()
        labels = [self.get_label(os.path.join(self.data_root, image)) for image in images]
        labels_number = {}
        for label in labels:
            if label in labels_number.keys():
                labels_number[label] += 1
            else:
                labels_number[label] = 1
        return labels_number

    def show_label_number_distr(self):
        """展示样本数目分布
        """
        labels_number = self.get_label_number()
        labels = labels_number.keys()
        number = labels_number.values()
        label_to_name = self.get_label_to_name()
        name = [label_to_name[str(label)] for label in labels]
        
        font = FontProperties(fname=r"font/simhei.ttf", size=7)
        ax1 = plt.subplot(111)
        x_axis = range(len(labels))
        rects = ax1.bar(x=x_axis, height=number, width=0.8, label='Label Number')
        plt.ylabel('Number')
        plt.xticks([index + 0.13 for index in x_axis], name, fontproperties=font, rotation=270)
        plt.xlabel('Labels')
        plt.title('Sample Number of Each Label')
        plt.legend()

        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()        
        plt.show()
            
    def get_images(self):
        """得到所用样本名称
        """
        files = os.listdir(self.data_root)
        images = list(filter(lambda x: x.endswith('jpg'), files))
        return images

    def get_label(self, image_name):
        """得到图片对应的类标

        Args:
            image_name: 图片名称
        Returns:
            label: 类标
        """
        label_txt = image_name.replace('jpg', 'txt')
        with open(label_txt, 'r') as label_file:
            for image_label in label_file:
                label = int(image_label.split(', ')[1])
        return label

    def get_label_to_name(self):
        """得到类别到真实名称的映射
        """
        with open(self.label_id_json, 'r') as f:
            label_to_name = json.load(f)
        return label_to_name


if __name__ == '__main__':
    data_root = 'data/huawei_data/train_data'
    label_id_json = 'data/huawei_data/label_id_name.json'
    dataset_statistic = DatasetStatistic(data_root, label_id_json)
    dataset_statistic.show_label_number_distr()
    pass
