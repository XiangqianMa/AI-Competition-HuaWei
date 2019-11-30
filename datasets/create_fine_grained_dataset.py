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
    def __init__(self, data_root, sample_list, coarse_label_list, fine_label_list, size, mean, std, transforms=None):
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
        self.coarse_label_list = coarse_label_list
        self.fine_label_list = fine_label_list
        self.size = size
        self.mean = mean
        self.std = std
        self.transforms = transforms
    
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
        coarse_grained_label = self.coarse_label_list[index]
        fine_grained_label = self.fine_label_list[index]
        if self.transforms:
            image = np.asarray(image)
            image = self.transforms(image)
            image = Image.fromarray(image)
        
        transform_train_list = [
                    T.Resize([300, 300], interpolation=3),
                    T.CenterCrop(self.size),
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std)
                ]          
        transform_compose = T.Compose(transform_train_list)
        image = transform_compose(image)
        coarse_grained_label = torch.tensor(coarse_grained_label).long()
        fine_grained_label = torch.tensor(fine_grained_label).long()

        return image, coarse_grained_label, fine_grained_label

    def __len__(self):
        """ 得到训练数据集总共有多少个样本
        """
        return len(self.sample_list)
    

class ValDataset(Dataset):
    def __init__(self, data_root, sample_list, coarse_label_list, fine_label_list, size, mean, std):
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
        self.coarse_grained_list = coarse_label_list
        self.fine_grained_list = fine_label_list
        self.size = size
        self.mean = mean
        self.std = std
    
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
        coarse_grained_label = self.coarse_grained_list[index]
        fine_grained_label = self.fine_grained_list[index]
        transform_val_list = [ 
                    T.Resize(self.size, interpolation=3),
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std)
                ]          
        transform_compose = T.Compose(transform_val_list)
        image = transform_compose(image)
        coarse_grained_label = torch.tensor(coarse_grained_label).long()
        fine_grained_label = torch.tensor(fine_grained_label).long()

        return image, coarse_grained_label, fine_grained_label

    def __len__(self):
        """ 得到训练数据集总共有多少个样本
        """
        return len(self.sample_list)


class GetDataloader(object):
    def __init__(self, data_root, folds_split=1, test_size=None, label_names_path='data/huawei_data/label_id_name.json'):
        """
        Args:
            data_root: str, 数据集根目录
            folds_split: int, 划分为几折
            test_size: 验证集占的比例, [0, 1]
        """
        self.data_root = data_root
        self.folds_split = folds_split
        self.test_size = test_size
        with open(label_names_path, 'r') as f:
            self.label_to_name = json.load(f)
        coarse_labels_name = sorted(set([coarse_label.split('/')[0] for coarse_label in self.label_to_name.values()]))
        self.coarse_labels_to_id = {coarse_label: index for index, coarse_label in enumerate(coarse_labels_name)}
        self.samples, self.coarse_grained_labels, self.fine_grained_labels = self.get_samples_labels()

        if folds_split == 1:
            if not test_size:
                raise ValueError('You must specified test_size when folds_split equal to 1.')
    
    def get_dataloader(self, batch_size, image_size, mean, std, transforms=None):
        """得到数据加载器
        Args:
            batch_size: int, 批量大小
            image_size: [height, width], 图片大小
            mean: tuple, 通道均值
            std: tuple, 通道方差
            transforms: callable, 数据增强方式
        Return:
            train_dataloader_folds: list, [train_dataloader_0, train_dataloader_1,...]
            valid_dataloader_folds: list, [val_dataloader_0, val_dataloader_1, ...]
        """
        train_lists, val_lists = self.get_split()
        train_dataloader_folds, valid_dataloader_folds = list(), list()
        self.draw_train_val_distribution(train_lists, val_lists)

        for train_list, val_list in zip(train_lists, val_lists):
            train_dataset = TrainDataset(self.data_root, train_list[0], train_list[1], train_list[2], image_size,
                                         transforms=transforms, mean=mean, std=std)
            val_dataset = ValDataset(self.data_root, val_list[0], val_list[1], val_list[2], image_size, mean=mean, std=std)

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
        train_coarse_grained_labels = [self.coarse_grained_labels[i] for i in train_index]
        train_fine_grained_labels = [self.fine_grained_labels[i] for i in train_index]
        
        val_samples = [self.samples[i] for i in val_index]
        val_coarse_grained_labels = [self.coarse_grained_labels[i] for i in val_index]
        val_fine_grained_labels = [self.fine_grained_labels[i] for i in val_index]

        return [[train_samples, train_coarse_grained_labels, train_fine_grained_labels]], [[val_samples, val_coarse_grained_labels, val_fine_grained_labels]]
    
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
            train_coarse_grained_labels = ([self.coarse_grained_labels[i] for i in train_index])
            train_fine_grained_labels = ([self.fine_grained_labels[i] for i in train_index])
            val_samples = ([self.samples[i] for i in val_index])
            val_coarse_grained_labels = ([self.coarse_grained_labels[i] for i in val_index])
            val_fine_grained_labels = ([self.fine_grained_labels[i] for i in val_index])
            train_folds.append([train_samples, train_coarse_grained_labels, train_fine_grained_labels])
            val_folds.append([val_samples, val_coarse_grained_labels, val_fine_grained_labels])
        return train_folds, val_folds

    def get_samples_labels(self):
        """ 得到所有的图片名称以及对应的类标
        Returns:
            samples: list, 所有的图片名称
            coarse_grained_labels: list, 所有的图片对应的大类类标, 和samples一一对应
            fine_grained_labels: list, 所有图片对应的小类类标，和samples一一对应
        """
        files_list = os.listdir(self.data_root)
        # 过滤得到标注文件
        annotations_files_list = [f for f in files_list if f.split('.')[1] == 'txt']

        samples = []
        fine_grained_labels = []
        coarse_grained_labels = []
        for annotation_file in annotations_files_list:
            annotation_file_path = os.path.join(self.data_root, annotation_file)
            with open(annotation_file_path, encoding='utf-8-sig') as f:
                for sample_label in f:
                    sample_name = sample_label.split(', ')[0]
                    fine_grained_label = int(sample_label.split(', ')[1])
                    coarse_grained_label = self.coarse_labels_to_id[self.label_to_name[str(fine_grained_label)].split('/')[0]]
                    samples.append(sample_name)
                    coarse_grained_labels.append(coarse_grained_label)
                    fine_grained_labels.append(fine_grained_label)
                    
        return samples, coarse_grained_labels, fine_grained_labels


if __name__ == "__main__":
    data_root = 'data/huawei_data/train_data'
    folds_split = 1
    test_size = 0.2
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    get_dataloader = GetDataloader(data_root, folds_split=1, test_size=test_size)
    train_list, val_list = get_dataloader.get_split()
    train_dataset = TrainDataset(data_root, train_list[0][0], train_list[0][1], train_list[0][2], size=[224, 224], mean=mean, std=std)
    for i in range(len(train_dataset)):
        image, coarse_label, fine_label = train_dataset[i]
    pass
