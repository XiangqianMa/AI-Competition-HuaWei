import torch
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class TrainDataset(Dataset):
    def __init__(self, data_root, sample_list, label_list, size, mean, std, transforms=None, only_self=False, only_official=False, multi_scale=False):
        """
        Args:
            data_root: str, 数据集根目录
            sample_list: list, 样本名
            label_list: list, 类标, 与sample_list中的样本按照顺序对应
            size: [height, width], 图片的目标大小
            mean: tuple, 通道均值
            std: tuple, 通道方差
            transforms: callable, 数据集转换方式
        """
        super(TrainDataset, self).__init__()
        self.data_root = data_root
        self.sample_list = sample_list
        self.label_list = label_list
        if only_self and only_official:
            raise ValueError('only_self, only_official should not be the same.')
        if only_official:
            sample_list = []
            label_list = []
            for sample, label in zip(self.sample_list, self.label_list):
                if 'img' in sample:
                    sample_list.append(sample)
                    label_list.append(label)
            self.sample_list = sample_list
            self.label_list = label_list
        if only_self:
            sample_list = []
            label_list = []
            for sample, label in zip(self.sample_list, self.label_list):
                if 'img' not in sample:
                    sample_list.append(sample)
                    label_list.append(label)
            self.sample_list = sample_list
            self.label_list = label_list            
        
        self.size = size
        self.mean = mean
        self.std = std
        self.transforms = transforms
        self.multi_scale = multi_scale
    
    def __getitem__(self, index):
        """
        Args:
            index: int, 当前的索引下标

        Returns:
            image: [channel, height, width] tensor, 当前索引下标对应的图像数据
            label: [1] tensor, 当前索引下标对应的图像数据对应的类标
        """
        sample_path = os.path.join(self.data_root, self.sample_list[index])
        image = Image.open(sample_path).convert('RGB')
        label = self.label_list[index]
        if self.transforms:
            image = np.asarray(image)
            image = self.transforms(image)
            image = Image.fromarray(image)
        
        # 如果不进行多尺度训练，则将图片转换为指定的图片大小，并转换为tensor
        if self.multi_scale:
            image = T.Resize(self.size, interpolation=3)(image)
            image = np.asarray(image)
        else:
            transform_train_list = [
                        T.Resize(self.size, interpolation=3),
                        T.ToTensor(),
                        T.Normalize(self.mean, self.std)
                    ]          
            transform_compose = T.Compose(transform_train_list)
            image = transform_compose(image)
        label = torch.tensor(label).long()

        return image, label

    def __len__(self):
        """ 得到训练数据集总共有多少个样本
        """
        return len(self.sample_list)
    

class ValDataset(Dataset):
    def __init__(self, data_root, sample_list, label_list, size, mean, std, only_self=False, only_official=False, multi_scale=False):
        """
        Args:
            data_root: str, 数据集根目录
            sample_list: list, 样本名
            label_list: list, 类标, 与sample_list中的样本按照顺序对应
            size: [height, width], 图片的目标大小
            mean: tuple, 通道均值
            std: tuple, 通道方差
        """
        super(ValDataset, self).__init__()
        self.data_root = data_root
        self.sample_list = sample_list
        self.label_list = label_list
        if only_self and only_official:
            raise ValueError('only_self, only_official should not be the same.')       
        if only_official:
            sample_list = []
            label_list = []
            for sample, label in zip(self.sample_list, self.label_list):
                if 'img' in sample:
                    sample_list.append(sample)
                    label_list.append(label)
            self.sample_list = sample_list
            self.label_list = label_list  
        if only_self:
            sample_list = []
            label_list = []
            for sample, label in zip(self.sample_list, self.label_list):
                if 'img' not in sample:
                    sample_list.append(sample)
                    label_list.append(label)
            self.sample_list = sample_list
            self.label_list = label_list              
        self.size = size
        self.mean = mean
        self.std = std
        self.multi_scale = multi_scale
    
    def __getitem__(self, index):
        """
        Args:
            index: int, 当前的索引下标

        Returns:
            image_name: str；图片名称
            image: [channel, height, width] tensor, 当前索引下标对应的图像数据
            label: [1] tensor, 当前索引下标对应的图像数据对应的类标
        """
        image_name = self.sample_list[index]
        sample_path = os.path.join(self.data_root, image_name)
        image = Image.open(sample_path).convert('RGB')
        label = self.label_list[index]
        
        if self.multi_scale:
            image = T.Resize(self.size, interpolation=3)(image)
        else:
            transform_val_list = [ 
                        T.Resize(self.size, interpolation=3),
                        T.ToTensor(),
                        T.Normalize(self.mean, self.std)
                    ]          
            transform_compose = T.Compose(transform_val_list)
            image = transform_compose(image)
        label = torch.tensor(label).long()

        return image_name, image, label

    def __len__(self):
        """ 得到训练数据集总共有多少个样本
        """
        return len(self.sample_list)


class GetDataloader(object):
    def __init__(self, data_root, folds_split=1, test_size=None, label_names_path='data/huawei_data/label_id_name.json', only_self=False, only_official=False, selected_labels=None):
        """
        Args:
            data_root: str, 数据集根目录
            folds_split: int, 划分为几折
            test_size: 验证集占的比例, [0, 1]
            selected_labels: list，被选中用于训练的类别
        """
        self.data_root = data_root
        self.folds_split = folds_split
        self.selected_labels = selected_labels
        if self.selected_labels:
            print('Selected Labels: ', self.selected_labels)
        with open(label_names_path, 'r') as f:
            self.label_to_name = json.load(f)

        self.samples, self.labels = self.get_samples_labels()
        self.test_size = test_size
        self.only_self = only_self
        self.only_official = only_official

        if folds_split == 1:
            if not test_size:
                raise ValueError('You must specified test_size when folds_split equal to 1.')
    
    def get_dataloader(self, batch_size, image_size, mean, std, transforms=None, multi_scale=False):
        """得到数据加载器
        Args:
            batch_size: int, 批量大小
            image_size: [height, width], 图片大小
            mean: tuple, 通道均值
            std: tuple, 通道方差
            transforms: callable, 数据增强方式
            multi_scale: 是否使用多尺度训练
        Return:
            train_dataloader_folds: list, [train_dataloader_0, train_dataloader_1,...]
            valid_dataloader_folds: list, [val_dataloader_0, val_dataloader_1, ...]
        """
        train_lists, val_lists = self.get_split()
        train_dataloader_folds, valid_dataloader_folds = list(), list()
        self.draw_train_val_distribution(train_lists, val_lists)

        for train_list, val_list in zip(train_lists, val_lists):
            train_dataset = TrainDataset(
                self.data_root, 
                train_list[0], 
                train_list[1], 
                image_size,
                transforms=transforms, 
                mean=mean, 
                std=std, 
                only_self=self.only_self, 
                only_official=self.only_official, 
                multi_scale=multi_scale
                )
            # 默认不在验证集上进行多尺度
            val_dataset = ValDataset(
                self.data_root, 
                val_list[0], 
                val_list[1], 
                image_size, 
                mean=mean, 
                std=std, 
                only_self=self.only_self, 
                only_official=self.only_official, 
                multi_scale=False
                )

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=8,
                pin_memory=True,
                shuffle=True
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=8,
                pin_memory=True,
                shuffle=False
            )
            train_dataloader_folds.append(train_dataloader)
            valid_dataloader_folds.append(val_dataloader)
        return train_dataloader_folds, valid_dataloader_folds

    def draw_train_val_distribution(self, train_lists, val_lists):
        """ 画出各个折的训练集与验证集的数据分布

        Args:
            train_lists: list, 每一个数据均为[train_sample, train_label], train_sample: list, 样本名称， train_label: list, 样本类标
            val_lists: list, 每一个数据均为[val_sample, val_label]， val_sample: list, 样本名称， val_label: list, 样本类标
        """
        for index, (train_list, val_list) in enumerate(zip(train_lists, val_lists)):
            train_labels_number = {}
            for label in train_list[1]:
                if label in train_labels_number.keys():
                    train_labels_number[label] += 1
                else:
                    train_labels_number[label] = 1
            self.draw_labels_number(train_labels_number, phase='Train_%s' % index)
            val_labels_number = {}
            for label in val_list[1]:
                if label in val_labels_number.keys():
                    val_labels_number[label] += 1
                else:
                    val_labels_number[label] = 1
            self.draw_labels_number(val_labels_number, phase='Val_%s' % index)

    def draw_labels_number(self, labels_number, phase='Train'):
        """ 画图函数
        Args:
            labels_number: dict, {label_1: number_1, label_2: number_2, ...}
            phase: str, 当前模式
        """
        labels = labels_number.keys()
        number = labels_number.values()
        name = [self.label_to_name[str(label)] for label in labels]
        
        plt.figure(figsize=(20, 16), dpi=240)
        font = FontProperties(fname=r"font/simhei.ttf", size=7)
        ax1 = plt.subplot(111)
        x_axis = range(len(labels))
        rects = ax1.bar(x=x_axis, height=number, width=0.8, label='Label Number')
        plt.ylabel('Number')
        plt.xticks([index + 0.13 for index in x_axis], name, fontproperties=font, rotation=270)
        plt.xlabel('Labels')
        plt.title('%s: Sample Number of Each Label' % phase)
        plt.legend()

        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
        plt.savefig('readme/%s.jpg' % phase, dpi=240)
        
    def get_split(self):
        """对数据集进行划分
        Return:
            train_list: list, 每一个数据均为[train_sample, train_label], train_sample: list, 样本名称， train_label: list, 样本类标
            val_list: list, 每一个数据均为[val_sample, val_label]， val_sample: list, 样本名称， val_label: list, 样本类标
        """
        if self.folds_split == 1:
            train_list, val_list = self.get_data_split_single()
        else:
            train_list, val_list = self.get_data_split_folds()

        return train_list, val_list
        
    def get_data_split_single(self):
        """随机划分训练集和验证集
        Return:
            [train_samples, train_labels], train_samples: list, 样本名称， train_labels: list, 样本类标
            [val_samples, val_labels], val_samples: list, 样本名称， val_labels: list, 样本类标
        """
        samples_index = [i for i in range(len(self.samples))]
        train_index, val_index = train_test_split(samples_index, test_size=self.test_size, random_state=69)
        train_samples = [self.samples[i] for i in train_index]
        train_labels = [self.labels[i] for i in train_index]
        val_samples = [self.samples[i] for i in val_index]
        val_labels = [self.labels[i] for i in val_index]
        return [[train_samples, train_labels]], [[val_samples, val_labels]]
    
    def get_data_split_folds(self):
        """交叉验证的数据划分
        Return:
            train_folds: list, 所有折的[train_samples, train_labels], train_samples: list, 样本名称， train_labels: list, 样本类标
            val_folds: list, 所有折的[val_samples, val_labels], val_samples: list, 样本名称， val_labels: list, 样本类标
        """
        skf = StratifiedKFold(n_splits=self.folds_split, shuffle=True, random_state=69)
        train_folds = []
        val_folds = []
        for train_index, val_index in skf.split(self.samples, self.labels):
            train_samples = ([self.samples[i] for i in train_index])
            train_labels = ([self.labels[i] for i in train_index])
            val_samples = ([self.samples[i] for i in val_index])
            val_labels = ([self.labels[i] for i in val_index])
            train_folds.append([train_samples, train_labels])
            val_folds.append([val_samples, val_labels])
        return train_folds, val_folds

    def get_samples_labels(self):
        """ 得到所有的图片名称以及对应的类标
        Returns:
            samples: list, 所有的图片名称
            labels: list, 所有的图片对应的类标, 和samples一一对应
        """
        files_list = os.listdir(self.data_root)
        # 过滤得到标注文件
        annotations_files_list = [f for f in files_list if f.split('.')[1] == 'txt']

        samples = []
        labels = []
        for annotation_file in annotations_files_list:
            annotation_file_path = os.path.join(self.data_root, annotation_file)
            with open(annotation_file_path, encoding='utf-8-sig') as f:
                for sample_label in f:
                    sample_name = sample_label.split(', ')[0]
                    label = int(sample_label.split(', ')[1])
                    if self.selected_labels:
                        # 依据父类别进行过滤
                        parent_label = self.label_to_name[str(label)].split('/')[0]
                        if parent_label in self.selected_labels:
                            samples.append(sample_name)
                            labels.append(label) 
                    else:                           
                        samples.append(sample_name)
                        labels.append(label)
        return samples, labels


def multi_scale_transforms(image_size, images, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    transform_train_list = [
                T.Resize(image_size, interpolation=3),
                T.ToTensor(),
                T.Normalize(mean, std)
            ]
    transform_compose = T.Compose(transform_train_list)
    images = images.numpy()
    images_resize = torch.zeros(images.shape[0], 3, image_size[0], image_size[1])
    for index in range(images.shape[0]):
        image = transform_compose(Image.fromarray(images[index]))
        images_resize[index] = image

    return images_resize
    

if __name__ == "__main__":
    data_root = 'data/huawei_data/train_data'
    folds_split = 1
    test_size = 0.2
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    get_dataloader = GetDataloader(data_root, folds_split=1, test_size=test_size)
    train_list, val_list = get_dataloader.get_split()
    train_dataset = TrainDataset(data_root, train_list[0], train_list[1], size=[224, 224], mean=mean, std=std)
    for i in range(len(train_dataset)):
        image, label = train_dataset[i]
    pass
