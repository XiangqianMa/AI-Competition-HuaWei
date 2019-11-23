# -*- coding:utf-8 -*-
import re
import requests
import os
import shutil
import random
import time

from utils.data_analysis import DatasetStatistic


class DownloadImages(object):
    def __init__(self, url, words, numbers, save_path):
        self.url = url
        self.words = words
        self.numbers = numbers
        self.save_path = save_path
        self.__prepare()

    def download_pics(self):
        for word, number in zip(self.words, self.numbers):
            word_url = self.url % word
            result = requests.get(word_url)
            self.dowmload_pic(result.text, word, number)

    def dowmload_pic(self, html, keyword, number):
        pic_url = re.findall('"objURL":"(.*?)",', html, re.S)
        i = 1
        print('找到关键词:' + keyword + '的图片，现在开始下载图片...')
        pic_url_selected = random.sample(pic_url, min(len(pic_url), number))
        for each in pic_url_selected:
            print('正在下载第' + str(i) + '张图片，图片地址:' + str(each))
            try:
                pic = requests.get(each, timeout=10)
            except:
                print('【错误】当前图片无法下载')
                continue

            dir = os.path.join(self.save_path, keyword + '_' + str(i) + '.jpg')
            fp = open(dir, 'wb')
            fp.write(pic.content)
            fp.close()
            i += 1
            time.sleep(0.2)

    def __prepare(self):
        if not os.path.exists(self.save_path):
            print('Making Dir: %s' % self.save_path)
            os.mkdir(self.save_path)
        else:
            print('%s exists, removing %s' % (self.save_path, self.save_path))
            shutil.rmtree(self.save_path)
            print('Making Dir: %s' % self.save_path)
            os.mkdir(self.save_path)


if __name__ == '__main__':
    data_root = 'data/huawei_data/train_data'
    label_id_json = 'data/huawei_data/label_id_name.json'
    dataset_statistic = DatasetStatistic(data_root, label_id_json)
    name_to_download_number = dataset_statistic.get_download_number()    
    words = name_to_download_number.keys()
    numbers = name_to_download_number.values()
    save_path = '/media/mxq/data/competition/HuaWei/download_images'
    url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + '%s' + '&ct=201326592&v=flip'
    downlaod_pic = DownloadImages(url, words, numbers, save_path)
    downlaod_pic.download_pics()
