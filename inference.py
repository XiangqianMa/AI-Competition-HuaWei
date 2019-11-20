# -*- coding: utf-8 -*-
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from model_service.pytorch_model_service import PTServingBaseService

import time
import json
from models.build_model import PrepareModel


class ImageClassificationService(PTServingBaseService):
    def __init__(self, model_name, model_path, label_json_path=None):
        """在服务器上进行前向推理，得到结果
        Args:

        """
        self.model_name = model_name
        self.model_path = model_path
        self.classes_num = 54
        
        self.use_cuda = False
        self.model, self.label_id_name_dict = self.__prepare__(label_json_path)
        self.model.eval()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )

        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize
        ])

    def inference(self, data):
        """
        Wrapper function to run preprocess, inference and postprocess functions.

        Parameters
        ----------
        data : map of object
            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        """
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()

        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        # 对数据进行推理
        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000

        data = self._postprocess(data)
        post_time_in_ms = (time.time() - infer_end_time) * 1000

        data['latency_time'] = pre_time_in_ms + infer_in_ms + post_time_in_ms
        return data
    
    def __prepare__(self, label_json_path):
        """准备模型，得到id到真实类别的映射
        """
        prepare_model = PrepareModel()
        model = prepare_model.create_model(self.model_name, self.classes_num)
        
        if torch.cuda.is_available():
            print('Using GPU for inference')
            self.use_cuda = True
            checkpoint = torch.load(self.model_path)
            model.load_state_dict(checkpoint['state_dict'])
            model = torch.nn.DataParallel(model).cuda()
        else:
            print('Using CPU for inference')
            checkpoint = torch.load(self.model_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
        # 得到类标到真实标注的映射
        with open(label_json_path, 'r') as f:
            label_dict = json.load(f)

        return model, label_dict
    
    def _inference(self, data):
        # 对单张样本得到预测结果
        img = data["input_img"]
        img = img.unsqueeze(0)
        if self.use_cuda:
            img = img.cuda()
        with torch.no_grad():
            pred_score = self.model(img)
            pred_score = F.softmax(pred_score.data, dim=1)
            if pred_score is not None:
                pred_label = torch.argsort(pred_score[0], descending=True)[:1][0].item()
                result = {'result': self.label_id_name_dict[str(pred_label)]}
            else:
                result = {'result': 'predict score is None'}

        return result
    
    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for _, file_content in v.items():
                img = Image.open(file_content)
                img = self.transforms(img)
                preprocessed_data[k] = img
        return preprocessed_data
    
    def _postprocess(self, data):
        return data
    