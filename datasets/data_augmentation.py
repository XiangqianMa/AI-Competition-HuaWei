import math
import random
import numpy as np
import torchvision.transforms as T
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, CLAHE, RandomRotate90, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, OneOf,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightnessContrast, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,Cutout,Rotate, Normalize, Crop, RandomCrop, Resize, RGBShift
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


class DataAugmentation(object):
    def __init__(self, erase_prob=0.0, full_aug=True, gray_prob=0.0):
        """
        Args:
            full_aug: 是否对整幅图片进行随机增强
        """
        self.full_aug = full_aug
        self.erase_prob = erase_prob
        self.gray_prob = gray_prob
        
        self.random_erase = RandomErasing(probability=erase_prob)
        self.rgb2gray = RGB2GRAY(p=gray_prob)

    def __call__(self, image):
        """
        :param image: 传入的图片
        :return: 经过数据增强后的图片
        """
        # 先随机擦除
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
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.25),
            ShiftScaleRotate(shift_limit=0.07, rotate_limit=10, p=0.4),
        ])
        
        augmented = augmentations(image=original_image)
        image_aug = augmented['image']

        return image_aug


if __name__ == "__main__":
    image_path = 'data/huawei_data/train_data'
    # augment = DataAugmentation(erase_flag=True, full_aug=True, gray=True)
    augment = DataAugmentation(erase_prob=1.0, gray_prob=1.0)
    images_name = [f for f in os.listdir(image_path) if f.endswith('jpg')]
    for image_name in images_name:
        plt.figure()
        image = Image.open(os.path.join(image_path, image_name)).convert('RGB')
        image = np.asarray(image)
        augmented = augment(image=image)

        plt.imshow(augmented)
        plt.show()
