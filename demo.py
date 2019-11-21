import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import os
import json
import tqdm
from PIL import Image, ImageFont, ImageDraw
from models.build_model import PrepareModel
from config import get_classify_config


class DemoResults(object):
    def __init__(self, model_type, classes_num, weight_path, image_size, label_json_path, mean=[], std=[]):
        self.model_type = model_type
        self.classes_num = classes_num
        self.weight_path = weight_path
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.model, self.label_dict = self.__prepare__(label_json_path)

    def predict_multi_smaples(self, samples_root, rank=1, show=False, save=False, save_path=''):
        samples_list = os.listdir(samples_root)
        samples_list = set([sample.split('.')[0] for sample in samples_list])
        images_name = [sample + '.jpg' for sample in samples_list]
        predict_results = []
        tbar = tqdm.tqdm(images_name)
        for image_name in tbar:
            image_path = os.path.join(samples_root, image_name)
            indexs, label_index, label, predict_label = self.predict_single_sample(image_path, rank=rank, show=show, save=save, save_path=save_path)
            predict_results.extend([image_name, indexs, label_index, label, predict_label])
        return predict_results

    def predict_single_sample(self, sample_path, rank=1, show=False, save=False, save_path=''):
        """对单张样本进行预测

        Args:
            sample_path: 样本路径
            rank: 返回前rank个预测结果
        Returns:
            indexs: 预测的类别序号
            label_index: 类别索引
            label: 真实类标，如：大雁塔
        """
        annotation_txt = sample_path.replace('jpg', 'txt')
        with open(annotation_txt, 'r') as f:
            for line in f:
                label_index = int(line.split(', ')[1])
        label = self.label_dict[str(label_index)]
        image = Image.open(sample_path).convert('RGB')
        original_image = image
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
        indexs = [indexs[-(i+1)] for i in range(rank)]
        predict_label = self.label_dict[str(indexs[0])]

        fontpath = "font/simhei.ttf"
        font = ImageFont.truetype(fontpath, 16)
        draw = ImageDraw.Draw(original_image)
        txt = '真实类标： ' + label
        draw.text((50, 50), txt, font=font, fill=(0, 0, 255))
        txt = '预测类标： ' + predict_label
        draw.text((50, 100), txt, font=font, fill=(0, 255, 0))
        if save:
            image_save_path = os.path.join(save_path, sample_path.split('/')[-1])
            plt.imsave(image_save_path, original_image)
        if show:
            plt.imshow(original_image)
            plt.show()
            # mng = plt.get_current_fig_manager()
            # mng.window.showMaximized()
        
        return indexs, label_index, label, predict_label

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
    weight_path = 'checkpoints/resnet50/log-2019-11-20T14-44-38/resnet50_fold0_best.pth'
    label_json_path = 'data/huawei_data/label_id_name.json'
    samples_root = 'data/demo_data/images'
    save_path = 'data/demo_data/results'
    rank = 1
    show = False
    save = True
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    demo_predicts = DemoResults(config.model_type, config.num_classes, weight_path, config.image_size, label_json_path, mean=mean, std=std)
    demo_predicts.predict_multi_smaples(samples_root, rank, show, save, save_path)