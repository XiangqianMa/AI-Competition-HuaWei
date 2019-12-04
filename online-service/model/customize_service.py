# -*- coding: utf-8 -*-
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from model_service.pytorch_model_service import PTServingBaseService
import time
from metric.metrics_manager import MetricsManager
import log
logger = log.getLogger(__name__)

from model.deploy_models.build_model import PrepareModel


class ImageClassificationService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        """在服务器上进行前向推理，得到结果
        Args:

        """
        logger.info('Creating ImageClassificationService')
        self.model_name = model_name
        self.model_path = model_path
        self.classes_num = 54

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
        
        self.model = self.__prepare()
        self.model.eval()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.transforms = transforms.Compose([
            transforms.Resize([256, 256]),
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

        if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)

        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000

        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)

        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        if self.model_name + '_LatencyInference' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)

        # Update overall latency metric
        if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)

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
            pred_score = self.model(img)
            pred_score = F.softmax(pred_score.data, dim=1)
            if pred_score is not None:
                pred_label = torch.argsort(pred_score[0], descending=True)[:1][0].item()
                result = {'result': self.label_id_name_dict[str(pred_label)]}
            else:
                result = {'result': 'predict score is None'}

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
