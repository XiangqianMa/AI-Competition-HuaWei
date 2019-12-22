import math
import random
import numpy as np
import json
import torchvision.transforms as T
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, CLAHE, RandomRotate90, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, OneOf,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightnessContrast, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, GaussNoise, Cutout, Rotate, Normalize, Crop, RandomCrop, Resize, RGBShift
)
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import copy
import time


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.15, r1=0.3, mean=(0.485, 0.456, 0.406)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        img = copy.deepcopy(img)
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.shape[0] * img.shape[1]

            # 计算采样面积和采样长宽比
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                image_roi = img[x1:x1 + h, y1:y1 + w, :]
                image_mean = np.mean(image_roi, axis=(0, 1))
                # R通道置零
                image_mean[0] = 0
                if img.shape[2] == 3:
                    img[x1:x1 + h, y1:y1 + w, 0] = image_mean[0]
                    img[x1:x1 + h, y1:y1 + w, 1] = image_mean[1]
                    img[x1:x1 + h, y1:y1 + w, 2] = image_mean[2]
                else:
                    img[x1:x1 + h, y1:y1 + w] = image_mean[0]
                return img

        return img


class RGB2GRAY(object):
    def __init__(self, p=0.5):
        self.probability = p

    def __call__(self, image):
        if random.uniform(0, 1) > self.probability:
            return image
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 合并为三通道，以输入网络
        image_gray = cv2.merge([image_gray, image_gray, image_gray])

        return image_gray


class ResizeEqualRatio(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        img = img.resize(self.size, self.interpolation)

        return img


class DataAugmentation(object):
    def __init__(self, erase_prob=0.0, full_aug=True, gray_prob=0.0):
        """
        Args:
            erase_prob: float, 随机擦除的概率
            full_aug: bool, 是否对图片进行随机增强
            gray_prob: float, 随机灰度变换的概率
        """
        self.full_aug = full_aug
        self.erase_prob = erase_prob
        self.gray_prob = gray_prob
        self.resize_equal_ratio = ResizeEqualRatio((int(256 * (256 / 224)), int(256 * (256 / 224))))
        self.random_erase = RandomErasing(probability=erase_prob)
        self.rgb2gray = RGB2GRAY(p=gray_prob)

    def __call__(self, image):
        """
        Args:
            image: array，传入的图片
        Returns:
            image: array，经过数据增强后的图片
        """
        # image = self.resize_equal_ratio(image)
        # image = np.asarray(image)
        # 随机擦除
        if self.erase_prob > 0:
            image = self.random_erase(image)
        # 转为灰度
        if self.gray_prob > 0:
            image = self.rgb2gray(image)
        if self.full_aug:
            image = self.data_augmentation(image)

        return image

    def data_augmentation(self, original_image):
        """ 进行样本和掩膜的随机增强
        Args:
            original_image: 原始图片
        Return:
            image_aug: 增强后的图片
        """
        augmentations = Compose([
            # CenterCrop(256, 256),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.25),
            ShiftScaleRotate(shift_limit=0.07, rotate_limit=10, p=0.4),
        ])

        augmented = augmentations(image=original_image)
        image_aug = augmented['image']

        return image_aug


if __name__ == "__main__":
    image_path = 'data/huawei_data/train_data'
    # 只显示特定类别
    show_classes = [33, 38]  # x for x in range(25, 54)
    augment = DataAugmentation(erase_prob=0, gray_prob=0)
    # 得到类标到真实标注的映射
    with open('data/huawei_data/label_id_name.json', 'r') as f:
        label_dict = json.load(f)

    images_name = [f for f in os.listdir(image_path) if f.endswith('jpg')]
    for image_name in images_name:
        # 打开txt文件，读取类别，若不在show_classes中，则跳过
        with open(os.path.join(image_path, image_name).replace('.jpg', '.txt'), 'r', encoding='utf-8-sig') as f:
            for line in f:
                txt_name = line.split(', ')[0]
                label_index = int(line.split(', ')[1])
        if label_index not in show_classes:
            continue

        plt.figure()
        image = Image.open(os.path.join(image_path, image_name)).convert('RGB')
        augmented = augment(image=image)
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Origin {}:{}'.format(image_name, label_dict[str(label_index)]))
        plt.subplot(1, 2, 2)
        plt.imshow(augmented)
        plt.title('Transform {}:{}'.format(image_name, label_dict[str(label_index)]))
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()