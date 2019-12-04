import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import os
import json
import tqdm
import shutil
from PIL import Image, ImageFont, ImageDraw
from models.build_model import PrepareModel
from config import get_classify_config
from datasets.create_dataset import GetDataloader


class DemoResults(object):
    def __init__(self, config, weight_path, label_json_path, fold, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        Args:
            config: 配置参数
            weight_path: str，权重文件的路径
            label_json_path: str, label_json_path文件的路径
            fold: int, 当前为第几折
            mean: tuple，各个通道的均值
            std: tuple，各个通道的方差
        """
        self.dataset_root = config.dataset_root
        self.model_type = config.model_type
        self.classes_num = config.num_classes
        self.last_stride = config.last_stride
        self.drop_rate = config.droprate
        self.image_size = config.image_size
        self.weight_path = weight_path
        self.fold = str(fold)
        self.mean = mean
        self.std = std
        self.model, self.label_dict = self.__prepare__(label_json_path)

    def predict_multi_smaples(self, valid_loader, rank=1, show=False, save=False, save_path=''):
        """
        Args:
            valid_loader: 验证集对应的Dataloader
            rank: int，返回前rank个预测结果
            show: bool, 是否显示图片
            save: bool, 是否保存图片
            save_path: str, 保存路径
        """
        tbar = tqdm.tqdm(valid_loader)
        with torch.no_grad():
            for i, (image_names, _, _) in enumerate(tbar):
                for image_name in image_names:
                    sample_path = os.path.join(self.dataset_root, image_name)
                    self.predict_single_sample(sample_path, rank, show, save, save_path)

    def predict_single_sample(self, sample_path, rank=1, show=False, save=False, save_path=''):
        """对单张样本进行预测

        Args:
            sample_path: str, 样本路径
            rank: int，返回前rank个预测结果
            show: bool, 是否显示图片
            save: bool, 是否保存图片
            save_path: str, 保存路径
        Returns:
            indexs: list，预测出的最相似的rank个类别索引
            label_index: int，真实类别索引
            predict_label: str, 预测出top1类标名称，如：大雁塔
            label: str，真实类标名称，如：大雁塔
        """
        annotation_txt = sample_path.replace('jpg', 'txt')
        with open(annotation_txt, 'r') as f:
            for line in f:
                label_index = int(line.split(', ')[1])
        label = self.label_dict[str(label_index)]
        image = Image.open(sample_path).convert('RGB')
        original_image = image.copy()
        transforms = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ])
        image = transforms(image)
        # 添加一个batch size通道
        image = torch.unsqueeze(image, dim=0).cuda()
        output = self.model(image)
        output = torch.squeeze(output)
        predicts = F.softmax(output, dim=0)
        predicts_numpy = predicts.cpu().detach().numpy()
        # 按行从小到大排列
        indexs = np.argsort(predicts_numpy)
        indexs = [indexs[-(i+1)] for i in range(rank)]
        predict_label = self.label_dict[str(indexs[0])]

        if indexs[0] == label_index:
            return

        fontpath = "font/simhei.ttf"
        font = ImageFont.truetype(fontpath, 16)
        draw = ImageDraw.Draw(original_image)
        txt = '真实类标： ' + label
        draw.text((50, 50), txt, font=font, fill=(0, 0, 255))
        txt = '预测类标： ' + predict_label
        draw.text((50, 100), txt, font=font, fill=(0, 255, 0))
        if save:
            plt.imshow(original_image)
            image_save_path = os.path.join(save_path, sample_path.split('/')[-1])
            original_image.save(image_save_path)
        if show:
            plt.show()
            # mng = plt.get_current_fig_manager()
            # mng.window.showMaximized()
        
        return indexs, label_index, predict_label, label

    def __prepare__(self, label_json_path):
        """
        Args:
            label_json_path: str, label_json_path文件的路径

        Returns:
            model: 加载训练好权重的模型
            label_dict: dict，类标名称与类标之间的对应关系
        """
        prepare_model = PrepareModel()
        model = prepare_model.create_model(self.model_type, self.classes_num, self.drop_rate, pretrained=False)
        model.load_state_dict(torch.load(self.weight_path)['state_dict'])
        print('Successfully Loaded from %s' % self.weight_path)
        model = model.cuda()
        model.eval()

        # 得到类标到真实标注的映射
        with open(label_json_path, 'r') as f:
            label_dict = json.load(f)

        return model, label_dict


if __name__ == "__main__":
    config = get_classify_config()
    model_type = config.model_type
    data_root = config.dataset_root
    folds_split = config.n_splits
    test_size = config.val_size
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms = None

    weight_path = os.path.join('checkpoints', model_type)
    lists = os.listdir(weight_path)  # 获得文件夹内所有文件
    lists.sort(key=lambda fn: os.path.getmtime(weight_path + '/' + fn))  # 排序
    weight_path = os.path.join(weight_path, lists[-1], 'model_best.pth')

    # 先删除该目录下所有的文件，再建立该文件夹
    save_path = 'data/demo_data/results'
    shutil.rmtree(save_path)
    os.makedirs(save_path)

    get_dataloader = GetDataloader(data_root, folds_split=folds_split, test_size=test_size)
    train_dataloaders, val_dataloaders = get_dataloader.get_dataloader(config.batch_size, config.image_size, mean, std,
                                                                       transforms=transforms)

    for fold_index, [train_loader, valid_loader] in enumerate(zip(train_dataloaders, val_dataloaders)):
        if fold_index in config.selected_fold:
            demo_predicts = DemoResults(
                config,
                weight_path,
                label_json_path='data/huawei_data/label_id_name.json',
                fold=fold_index,
                mean=mean,
                std=std
            )
            demo_predicts.predict_multi_smaples(valid_loader, rank=1, show=False, save=True, save_path=save_path)
