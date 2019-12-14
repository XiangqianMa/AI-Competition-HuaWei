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


#############################################
# 进行伪标签预测，并将大于设定阈值的样本移动到指定目录
#############################################
class PredictDownloadImage(object):
    def __init__(self, model_type, classes_num, weight_path, image_size, label_json_path, mean=[], std=[]):
        self.model_type = model_type
        self.classes_num = classes_num
        self.weight_path = weight_path
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.model, self.label_dict = self.__prepare__(label_json_path)

    def predict_multi_smaples(self, samples_root, thresh={}, save_path=''):
        """预测多张样本的伪标签，并将保留下的样本存放至指定的目录下

        Args:
            samples_root: 原始样本的路径
            thresh: dir, {'大雁塔': 0.95, ...}
            save_path: 保存路径
        """
        samples_list = os.listdir(samples_root)
        samples_list = set([sample.split('.')[0] for sample in samples_list])
        images_name = [sample + '.jpg' for sample in samples_list]
        predict_results = []
        tbar = tqdm.tqdm(images_name)
        if not os.path.exists(save_path):
            print('Making %s' % save_path)
            os.mkdir(save_path)
        else:
            print('Removing %s' % save_path)
            shutil.rmtree(save_path)
            print('Making %s' % save_path)
            os.mkdir(save_path)
        for image_name in tbar:
            label = image_name.split('_')[0]
            if label == '浆水鱼鱼':
                label = '凉鱼'
            elif label == '酥饺':
                label = '蜜饯张口酥饺'
            current_thresh = thresh[label]
            image_path = os.path.join(samples_root, image_name)
            index, predict_label, remain = self.predict_single_sample(label, image_path, thresh=current_thresh)
            if remain:
                descript = 'Remain: %s' % image_name
                self.save_image_label(save_path, image_path, image_name, predict_label, index)
            else:
                descript = 'Removing: %s' % image_name
            tbar.set_description(desc=descript)
        return predict_results

    def predict_single_sample(self, annotation, sample_path, thresh=0.6):
        """对单张样本进行预测

        Args:
            annotation: 标注的标签
            sample_path: 样本路径
            rank: 返回前rank个预测结果
        Returns:
            index: 预测的类别标号
            label: 真实类标，如：大雁塔
            remain: bool, True: 保留， False: 不保留
        """
        try:
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
            # 得分大于阈值且预测出的标签和标注的标签相同时保留
            if score > thresh and predict_label.split('/')[1] == annotation:
                remain = True
            else:
                remain = False
        except:
            remain = False
            index = -1
            predict_label = -1

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
        model = prepare_model.create_model(self.model_type, self.classes_num, 0, pretrained=False)
        model.load_state_dict(torch.load(self.weight_path)['state_dict'])
        model = model.cuda()
        model.eval()

        # 得到类标到真实标注的映射
        with open(label_json_path, 'r') as f:
            label_dict = json.load(f)

        return model, label_dict


def compute_labels_thresh(labels_scores, thresh_max=0.95, thresh_min=0.85):
    """依据各个类别的分数计算产生伪标签时的阈值

    Args:
        labels_scores: dir, {'大雁塔': 0.85, ...}
        thresh_max: 最大阈值
        thresh_min: 最小阈值
    Returns:
        labels_thresh: 类别对应的阈值 dir, {’大雁塔': 0.85, ...}
    """
    scores = labels_scores.values()
    max_score = max(scores)
    min_score = min(scores)
    labels_thresh = {}
    for label, score in labels_scores.items():
        thresh = (max_score - score) / (max_score - min_score) * (thresh_max - thresh_min) + thresh_min
        labels_thresh[label.split('/')[1]] = thresh
    
    return labels_thresh


if __name__ == "__main__":
    config = get_classify_config()
    weight_path = 'checkpoints/se_resnext101_32x4d/log-2019-12-08T22-39-47-0.9740/model_best.pth'
    label_json_path = 'data/huawei_data/label_id_name.json'
    samples_root = '/media/mxq/data/competition/HuaWei/下载的图片/补充'
    save_path = '/media/mxq/data/competition/HuaWei/下载的图片/psudeo_image_补充'
    labels_score_file = 'checkpoints/se_resnext101_32x4d/log-2019-12-08T22-39-47-0.9740/classes_acc.json'

    thresh_max = 0.95
    thresh_min = 0.95
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    with open(labels_score_file, 'r') as f:
        labels_score = json.load(f)
    labels_thresh = compute_labels_thresh(labels_score, thresh_max, thresh_min)
    print(labels_thresh)
    predict_download_images = PredictDownloadImage(config.model_type, config.num_classes, weight_path, config.image_size, label_json_path, mean=mean, std=std)
    predict_download_images.predict_multi_smaples(samples_root, thresh=labels_thresh, save_path=save_path)