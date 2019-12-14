import os
import json
import matplotlib.pyplot as plt
import random
import imagesize
from matplotlib.font_manager import FontProperties


class DatasetStatistic:
    """对数据集的分布进行统计
    """
    def __init__(self, data_root, label_id_json):
        """
        Args:
            data_root: str, 数据根目录
            label_id_json: str, label_id_json文件目录
        """
        self.data_root = data_root
        self.label_id_json = label_id_json
    
    def get_expand_number(self, thresh, more_than_thresh_number, less_than_thresh_number):
        """获取每一类别需要额外补充的样本数目

        Args:
            thresh: 样本数目阈值
            more_than_thresh_number: 大于thresh的类别的补充数目
            less_than_thresh_number: 小于thresh的类别的补充数目
        Returns:
            name_expand_number: dir, {'大雁塔‘： 10, ...}
        """
        labels_number = self.get_label_number()
        label_to_name = self.get_label_to_name()
        name_expand_number = {}
        for (label, number) in labels_number.items():
            if number > thresh or number == thresh:
                name_expand_number[str(label_to_name[str(label)])] = more_than_thresh_number
            else:
                name_expand_number[str(label_to_name[str(label)])] = (less_than_thresh_number - number) + random.sample([5, 10, 15], 1)[0]
        
        return name_expand_number

    def get_name_less_than_thresh(self, thresh):
        labels_number = self.get_label_number()
        label_to_name = self.get_label_to_name()
        names = []
        for label, number in labels_number.items():
            if number < thresh:
                names.append(label_to_name[str(label)])
        return names

    def get_label_number(self):
        """得到每一个类别对应的样本数目

        Returns:
            labels_number: dir {1: 256, 2:125, ...}
        """
        image_names = self.get_image_names()
        labels = [self.get_label(os.path.join(self.data_root, image_name)) for image_name in image_names]
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
            
    def get_image_names(self):
        """得到所用样本名称
        """
        files = os.listdir(self.data_root)
        images_names = list(filter(lambda x: x.endswith('jpg'), files))
        return images_names

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

    def show_image_aspect_ratio_distr(self):
        """得到样本长宽比
        """
        aspect_ratio_dict = {}

        image_names = self.get_image_names()
        for image_name in image_names:
            sample_path = os.path.join(self.data_root, image_name)
            width, height = imagesize.get(sample_path)
            aspect_ratio = width/height
            if aspect_ratio in aspect_ratio_dict:
                aspect_ratio_dict[aspect_ratio] += 1
            else:
                aspect_ratio_dict[aspect_ratio] = 0

        aspect_ratio_dict_filt = {}
        for key, value in aspect_ratio_dict.items():
            if value > 100:
                aspect_ratio_dict_filt[key] = value
        del aspect_ratio_dict
        plt.bar(aspect_ratio_dict_filt.keys(), aspect_ratio_dict_filt.values())
        plt.show()


if __name__ == '__main__':
    data_root = '/media/mxq/data/competition/HuaWei/下载的图片/combine_50pages'
    label_id_json = 'data/huawei_data/label_id_name.json'
    dataset_statistic = DatasetStatistic(data_root, label_id_json)
    dataset_statistic.show_label_number_distr()
    # names = dataset_statistic.get_name_less_than_thresh(100)
    # print(names)
