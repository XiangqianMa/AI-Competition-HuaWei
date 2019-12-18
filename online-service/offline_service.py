# -*- coding: utf-8 -*-
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
import logging
logger = logging.getLogger(__name__)
logger.info('from model.deploy_models.build_model import PrepareModel')

from model.deploy_models.build_model import PrepareModel


class ImageClassificationService:
    def __init__(self, model_name, model_path):
        """在服务器上进行前向推理，得到结果
        Args:

        """
        logger.info('Creating ImageClassificationService')
        self.model_name = model_name
        self.model_path = model_path
        self.classes_num = 59
        self.parent_classes = 5
        self.children_num_classes = [9, 14, 2, 6, 23]
        self.children_predicts_index = [[0, 9], [9, 23], [23, 25], [25, 31], [31, 54]]

        self.use_cuda = False
        self.label_id_name_dict = \
            {
                "0": "工艺品/仿唐三彩",
                "1": "工艺品/仿宋木叶盏",
                "2": "工艺品/布贴绣",
                "3": "工艺品/景泰蓝",
                "4": "工艺品/木马勺脸谱",
                "5": "工艺品/柳编",
                "6": "工艺品/葡萄花鸟纹银香囊",
                "7": "工艺品/西安剪纸",
                "8": "工艺品/陕历博唐妞系列",
                "9": "景点/关中书院",
                "10": "景点/兵马俑",
                "11": "景点/南五台",
                "12": "景点/大兴善寺",
                "13": "景点/大观楼",
                "14": "景点/大雁塔",
                "15": "景点/小雁塔",
                "16": "景点/未央宫城墙遗址",
                "17": "景点/水陆庵壁塑",
                "18": "景点/汉长安城遗址",
                "19": "景点/西安城墙",
                "20": "景点/钟楼",
                "21": "景点/长安华严寺",
                "22": "景点/阿房宫遗址",
                "23": "民俗/唢呐",
                "24": "民俗/皮影",
                "25": "特产/临潼火晶柿子",
                "26": "特产/山茱萸",
                "27": "特产/玉器",
                "28": "特产/阎良甜瓜",
                "29": "特产/陕北红小豆",
                "30": "特产/高陵冬枣",
                "31": "美食/八宝玫瑰镜糕",
                "32": "美食/凉皮",
                "33": "美食/凉鱼",
                "34": "美食/德懋恭水晶饼",
                "35": "美食/搅团",
                "36": "美食/枸杞炖银耳",
                "37": "美食/柿子饼",
                "38": "美食/浆水面",
                "39": "美食/灌汤包",
                "40": "美食/烧肘子",
                "41": "美食/石子饼",
                "42": "美食/神仙粉",
                "43": "美食/粉汤羊血",
                "44": "美食/羊肉泡馍",
                "45": "美食/肉夹馍",
                "46": "美食/荞面饸饹",
                "47": "美食/菠菜面",
                "48": "美食/蜂蜜凉粽子",
                "49": "美食/蜜饯张口酥饺",
                "50": "美食/西安油茶",
                "51": "美食/贵妃鸡翅",
                "52": "美食/醪糟",
                "53": "美食/金线油塔"
            }        
        
        self.parent_name_to_label, self.children_name_to_label = self.get_multi_name_to_label()
        # 类标到真实类别的转换
        self.parent_label_to_name, self.children_label_to_name = self.get_label_to_name()

        self.model = self.__prepare()
        self.model.eval()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.transforms = transforms.Compose([
            transforms.Resize([416, 416]),
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
        logger.info('At inference')
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()

        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')

        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000

        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)

        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')

        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = pre_time_in_ms + infer_in_ms + post_time_in_ms
        return data

    def _inference(self, data):
        """实际推理请求方法
        """
        logger.info('At _inference')

        # 对单张样本得到预测结果
        img = data["input_img"]
        img = img.unsqueeze(0)
        if self.use_cuda:
            img = img.cuda()
        with torch.no_grad():
            predict = self.model(img).squeeze()
            parent_predict = predict[:self.parent_classes]
            parent_scores = F.softmax(parent_predict.data)
            predict_parent_label = torch.argmax(parent_scores).item()
            start_index = self.children_predicts_index[predict_parent_label][0] + self.parent_classes
            end_index = self.children_predicts_index[predict_parent_label][1] + self.parent_classes
            child_predict = predict[start_index:end_index]
            child_scores = F.softmax(child_predict.data)
            predict_child_label = torch.argmax(child_scores).item()
        parent_name = self.parent_label_to_name[predict_parent_label]
        child_name = self.children_label_to_name[parent_name][predict_child_label]
        final_name = parent_name + '/' + child_name
        result = {'result': final_name}

        return result

    def __prepare(self):
        """准备模型
        """
        prepare_model = PrepareModel()
        model = prepare_model.create_model('se_resnext101_32x4d', self.classes_num, drop_rate=0, pretrained=False)

        if torch.cuda.is_available():
            logger.info('Using GPU for inference')
            self.use_cuda = True
            checkpoint = torch.load(self.model_path)
            model.load_state_dict(checkpoint['state_dict'])
            model = torch.nn.DataParallel(model).cuda()
        else:
            logger.info('Using CPU for inference')
            checkpoint = torch.load(self.model_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])

        return model

    def _preprocess(self, data):
        """预处理方法，在推理请求前调用，用于将API接口用户原始请求数据转换为模型期望输入数据
        """
        preprocessed_data = {}
        for k, v in data.items():
            for _, file_content in v.items():
                img = Image.open(file_content)
                img = self.transforms(img)
                preprocessed_data[k] = img
        return preprocessed_data

    def _postprocess(self, data):
        """后处理方法，在推理请求完成后调用，用于将模型输出转换为API接口输出
        """
        return data

    def get_label_to_name(self):
        """获得由类别转换为真实名称的字典
        
        Returns:
            parent_label_to_name: {0: ‘工艺品’，....}
            children_label_to_name: {'工艺品': {0: '仿唐三彩', ...}, ...}
        """
        # 父类的类别到名称
        parent_label_to_name = {}
        for name, label in self.parent_name_to_label.items():
            parent_label_to_name[label] = name
        # 子类的类别到名称
        children_label_to_name = {}
        for parent_label, children_name_to_label in self.children_name_to_label.items():
            label_to_name = {}
            for name, label in children_name_to_label.items():
                label_to_name[label] = name
            children_label_to_name[parent_label_to_name[parent_label]] = label_to_name
        return parent_label_to_name, children_label_to_name

    def get_multi_name_to_label(self):
        """多级真实类别到类标的映射

        Returns:
            parent_name_to_label: dir, {'工艺品': 0, '美食': 1, ...}
            parent_to_childern_label: dir, {'1': {'酥饺': 0, '凉鱼': 1, ....}, '工艺品': {'景泰蓝': 0, ...}, ...}
        """
        parent_to_childern = {}
        for label in self.label_id_name_dict.values():
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


if __name__ == "__main__":
    data = {}
    data['input_img'] = {'1': '/media/mxq/data/competition/HuaWei/train_data/img_4536.jpg'}
    model_path = '/media/mxq/project/Projects/competition/HuaWei/AI-Competition-HuaWei/checkpoints/se_resnext101_32x4d/model_best.pth'
    image_classify_service = ImageClassificationService('se_resnext101_32x4d', model_path)
    result = image_classify_service.inference(data)
    print(result)