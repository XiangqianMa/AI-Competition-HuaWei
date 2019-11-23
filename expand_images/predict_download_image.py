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


class PredictDownloadImage(object):
    def __init__(self, model_type, classes_num, weight_path, image_size, label_json_path, mean=[], std=[]):
        self.model_type = model_type
        self.classes_num = classes_num
        self.weight_path = weight_path
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.model, self.label_dict = self.__prepare__(label_json_path)

    def predict_multi_smaples(self, samples_root, thresh=0.6, save_path=''):
        samples_list = os.listdir(samples_root)
        samples_list = set([sample.split('.')[0] for sample in samples_list])
        images_name = [sample + '.jpg' for sample in samples_list]
        predict_results = []
        tbar = tqdm.tqdm(images_name)
        for image_name in tbar:
            image_path = os.path.join(samples_root, image_name)
            index, predict_label, remain = self.predict_single_sample(image_path, thresh=thresh)
            if remain:
                descript = 'Remain: %s' % image_name
                self.save_image_label(save_path, image_path, image_name, predict_label, index)
            else:
                descript = 'Removing: %s' % image_name
            tbar.set_description(desc=descript)
        return predict_results

    def predict_single_sample(self, sample_path, thresh=0.6):
        """对单张样本进行预测

        Args:
            sample_path: 样本路径
            rank: 返回前rank个预测结果
        Returns:
            index: 预测的类别标号
            label: 真实类标，如：大雁塔
            remain: bool, True: 保留， False: 不保留
        """
        image = Image.open(sample_path).convert('RGB')
        transforms = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ])
        image = transforms(image)
        image = torch.unsqueeze(image, dim=0).cuda()
        output = self.model(image)
        output = torch.squeeze(output)
        predicts = F.softmax(output)
        predicts_numpy = predicts.cpu().detach().numpy()
        indexs = np.argsort(predicts_numpy)
        index = indexs[-1]
        predict_label = self.label_dict[str(index)]
        score = predicts_numpy[index]
        if score > thresh:
            remain = True
        else:
            remain = False

        return index, predict_label, remain

    def save_image_label(self, save_path, image_path, image_name, label, index):
        """保存图片和类别文件

        Args:
            save_path: 保存根目录
            image_path: 原始图片路径
            image_name: 图片名称
            label: 真实类别名称
            index: 类别索引
        """
        label_file_name = image_name.replace('jpg', 'txt')
        label_file_path = os.path.join(save_path, label_file_name)
        with open(label_file_path, 'w') as f:
            line = image_name + ', ' + str(index)
            f.writelines(line)
        save_image_path = os.path.join(save_path, image_name)
        shutil.copy(image_path, save_image_path)

    def __prepare__(self, label_json_path):
        prepare_model = PrepareModel()
        model = prepare_model.create_model(self.model_type, self.classes_num)
        model.load_state_dict(torch.load(self.weight_path)['state_dict'])
        model = model.cuda()
        model.eval()

        # 得到类标到真实标注的映射
        with open(label_json_path, 'r') as f:
            label_dict = json.load(f)

        return model, label_dict


if __name__ == "__main__":
    config = get_classify_config()
    weight_path = 'checkpoints/resnet50/0.93/model_best.pth'
    label_json_path = 'data/huawei_data/label_id_name.json'
    samples_root = '/media/mxq/data/competition/HuaWei/download_images'
    save_path = '/media/mxq/data/competition/HuaWei/cleaned_dowload_images'
    thresh = 0.6
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    predict_download_images = PredictDownloadImage(config.model_type, config.num_classes, weight_path, config.image_size, label_json_path, mean=mean, std=std)
    predict_download_images.predict_multi_smaples(samples_root, thresh=thresh, save_path=save_path)