import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class TrainDataset(Dataset):
    def __init__(self, data_root, sample_list, label_list, size, mean, std, transforms=None):
        super(TrainDataset, self).__init__()
        self.data_root = data_root
        self.sample_list = sample_list
        self.label_list = label_list
        self.size = size
        self.mean = mean
        self.std = std
        self.transforms = transforms
    
    def __getitem__(self, index):
        sample_path = os.path.join(self.data_root, self.sample_list[index])
        image = Image.open(sample_path).convert('RGB')
        label = self.label_list[index]
        if self.transforms:
            image  = np.asarray(image)
            image = self.transforms(image)
            image = Image.fromarray(image)
        
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
        return len(self.sample_list)
    

class ValDataset(Dataset):
    def __init__(self, data_root, sample_list, label_list, size, mean, std):
        super(ValDataset, self).__init__()
        self.data_root = data_root
        self.sample_list = sample_list
        self.label_list = label_list
        self.size = size
        self.mean = mean
        self.std = std
    
    def __getitem__(self, index):
        sample_path = os.path.join(self.data_root, self.sample_list[index])
        image = Image.open(sample_path).convert('RGB')
        label = self.label_list[index]
        transform_val_list = [ 
                    T.Resize(self.size, interpolation=3),
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std)
                ]          
        transform_compose = T.Compose(transform_val_list)
        image = transform_compose(image)
        label = torch.tensor(label).long()

        return image, label

    def __len__(self):
        return len(self.sample_list)


class GetDataloader(object):
    def __init__(self, data_root, folds_split=1, test_size=None):
        """
        Args:
            data_root: 数据集根目录
            folds_split: int, 划分为几折
            test_size: 验证集占的比例, [0, 1]
        """
        self.data_root = data_root
        self.folds_split = folds_split
        self.samples, self.labels = self.get_samples_labels()
        self.test_size = test_size

        if folds_split == 1:
            if not test_size:
                raise ValueError('You must specified test_size when folds_split equal to 1.')
    
    def get_dataloader(self, batch_size, size, mean, std, transforms=None):
        train_lists, val_lists = self.get_split()
        train_dataloader_folds, valid_dataloader_folds = list(), list()
        
        for train_list, val_list in zip(train_lists, val_lists):
            train_dataset = TrainDataset(self.data_root, train_list[0], train_list[1], size, transforms=transforms, mean=mean, std=std)
            val_dataset = ValDataset(self.data_root, val_list[0], val_list[1], size, mean=mean, std=std)

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
        """
        Return:
            train_list: [train_sample, train_label], train_sample: list, 样本名称， train_label: list, 样本类标
            val_list: [val_sample, val_label]， val_sample: list, 样本名称， val_label: list, 样本类标
        """
        if self.folds_split == 1:
            train_list, val_list = self.get_data_split_single()
        else:
            train_list, val_list = self.get_data_split_folds()

        return train_list, val_list
        
    def get_data_split_single(self):
        samples_index = [i for i in range(len(self.samples))]
        train_index, val_index = train_test_split(samples_index, test_size=self.test_size, stratify=self.labels, random_state=69)
        train_samples = [self.samples[i] for i in train_index]
        train_labels = [self.labels[i] for i in train_index]
        val_samples = [self.samples[i] for i in val_index]
        val_labels = [self.labels[i] for i in val_index]
        return [[train_samples, train_labels]], [[val_samples, val_labels]]
    
    def get_data_split_folds(self):
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
        files_list = os.listdir(self.data_root)
        # 过滤得到标注文件
        annotations_files_list = [f for f in files_list if f.split('.')[1] == 'txt']

        samples = []
        labels = []
        for annotation_file in annotations_files_list:
            annotation_file_path = os.path.join(self.data_root, annotation_file)
            with open(annotation_file_path) as f:
                for sample_label in f:
                    sample_name = sample_label.split(', ')[0]
                    label = int(sample_label.split(', ')[1])
                    samples.append(sample_name)
                    labels.append(label)
        return samples, labels


if __name__ == "__main__":
    data_root = '/media/mxq/data/competition/HuaWei/train_data'
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
