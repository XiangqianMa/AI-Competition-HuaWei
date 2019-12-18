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
    def __init__(self, data_root, sample_list, parent_label_list, child_label_list, size, mean, std, transforms=None, multi_scale=False):
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
        self.parent_label_list = parent_label_list
        self.child_label_list = child_label_list

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
        parent_label = self.parent_label_list[index]
        child_label = self.child_label_list[index]
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
        parent_label = torch.tensor(parent_label).long()
        child_label = torch.tensor(child_label).long()

        return image, parent_label, child_label

    def __len__(self):
        """ 得到训练数据集总共有多少个样本
        """
        return len(self.sample_list)
    

class ValDataset(Dataset):
    def __init__(self, data_root, sample_list, parent_label_list, child_label_list, size, mean, std, multi_scale=False):
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
        self.parent_label_list = parent_label_list
        self.child_label_list = child_label_list

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
        parent_label = self.parent_label_list[index]
        child_label = self.child_label_list[index]
        
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
        parent_label = torch.tensor(parent_label).long()
        child_label = torch.tensor(child_label).long()

        return image_name, image, parent_label, child_label

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
        self.parent_name_to_label, self.parent_to_childern_label = self.get_multi_name_to_label()
        self.samples, self.parent_labels, self.child_labels = self.get_samples_labels()
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

        for train_list, val_list in zip(train_lists, val_lists):
            train_dataset = TrainDataset(
                self.data_root, 
                train_list[0], 
                train_list[1],
                train_list[2],
                image_size,
                transforms=transforms, 
                mean=mean, 
                std=std,
                multi_scale=multi_scale
                )
            # 默认不在验证集上进行多尺度
            val_dataset = ValDataset(
                self.data_root, 
                val_list[0], 
                val_list[1],
                val_list[2],
                image_size, 
                mean=mean, 
                std=std,
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
        train_parent_labels = [self.parent_labels[i] for i in train_index]
        train_child_labels = [self.child_labels[i] for i in train_index]
        val_samples = [self.samples[i] for i in val_index]
        val_parent_labels = [self.parent_labels[i] for i in val_index]
        val_child_labels = [self.child_labels[i] for i in val_index]
        return [[train_samples, train_parent_labels, train_child_labels]], [[val_samples, val_parent_labels, val_child_labels]]
    
    def get_data_split_folds(self):
        """交叉验证的数据划分
        Return:
            train_folds: list, 所有折的[train_samples, train_labels], train_samples: list, 样本名称， train_labels: list, 样本类标
            val_folds: list, 所有折的[val_samples, val_labels], val_samples: list, 样本名称， val_labels: list, 样本类标
        """
        skf = StratifiedKFold(n_splits=self.folds_split, shuffle=True, random_state=69)
        train_folds = []
        val_folds = []
        for train_index, val_index in skf.split(self.samples, self.child_labels):
            train_samples = ([self.samples[i] for i in train_index])
            train_parent_labels = [self.parent_labels[i] for i in train_index]
            train_child_labels = [self.child_labels[i] for i in train_index]
            val_samples = ([self.samples[i] for i in val_index])
            val_parent_labels = [self.parent_labels[i] for i in val_index]
            val_child_labels = [self.child_labels[i] for i in val_index]
            train_folds.append([train_samples, train_parent_labels, train_child_labels])
            val_folds.append([val_samples, val_parent_labels, val_child_labels])
        return train_folds, val_folds

    def get_samples_labels(self):
        """ 得到所有的图片名称以及对应的类标
        Returns:
            samples: list, 所有的图片名称
            parent_labels: list, 所有的图片对应的父类标, 和samples一一对应
            child_labels： list, 所有的图片对应的子类标, 和samples一一对应
        """
        files_list = os.listdir(self.data_root)
        # 过滤得到标注文件
        annotations_files_list = [f for f in files_list if f.split('.')[1] == 'txt']

        samples = []
        parent_labels = []
        child_labels = []
        for annotation_file in annotations_files_list:
            annotation_file_path = os.path.join(self.data_root, annotation_file)
            with open(annotation_file_path, encoding='utf-8-sig') as f:
                for sample_label in f:
                    sample_name = sample_label.split(', ')[0]
                    child_label = int(sample_label.split(', ')[1])
                    parent_name = self.label_to_name[str(child_label)].split('/')[0]
                    child_name = self.label_to_name[str(child_label)].split('/')[1]
                    if self.selected_labels:
                        # 依据父类别进行过滤
                        if parent_name in self.selected_labels:
                            samples.append(sample_name)
                            parent_label = self.parent_name_to_label[parent_name]
                            parent_labels.append(parent_label)
                            child_labels.append(self.parent_to_childern_label[parent_label][child_name])
                    else:
                        samples.append(sample_name)
                        parent_label = self.parent_name_to_label[parent_name]
                        parent_labels.append(parent_label)
                        child_labels.append(self.parent_to_childern_label[parent_label][child_name])
        return samples, parent_labels, child_labels

    def get_multi_name_to_label(self):
        """多级真实类别到类标的映射

        Returns:
            parent_name_to_label: dir, {'工艺品': 0, '美食': 1, ...}
            parent_to_childern_label: dir, {'1': {'酥饺': 0, '凉鱼': 1, ....}, '工艺品': {'景泰蓝': 0, ...}, ...}
        """
        parent_to_childern = {}
        for label in self.label_to_name.values():
            parent, children = label.split('/')
            if parent not in parent_to_childern.keys():
                parent_to_childern[parent] = [children]
            else:
                parent_to_childern[parent].append(children)
        parent_name = sorted(parent_to_childern.keys())
        parent_name_to_label = {name:index for index, name in enumerate(parent_name)}
        parent_to_childern_label = {}
        for parent_name, children_name in parent_to_childern.items():
            parent_label = parent_name_to_label[parent_name]
            children_name = sorted(children_name)
            children_name_to_label = {name:index for index, name in enumerate(children_name)}
            parent_to_childern_label[parent_label] = children_name_to_label
        return parent_name_to_label, parent_to_childern_label
        

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
    multi_label_name_to_label = get_dataloader.get_multi_name_to_label()
    print(multi_label_name_to_label)
    # train_list, val_list = get_dataloader.get_split()
    # train_dataset = TrainDataset(data_root, train_list[0], train_list[1], size=[224, 224], mean=mean, std=std)
    # for i in range(len(train_dataset)):
    #     image, label = train_dataset[i]
    # pass
