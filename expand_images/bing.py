# -*- coding: utf-8 -*-
"""根据搜索词下载百度图片"""
import re
import sys
import urllib

import requests
from utils.data_analysis import DatasetStatistic
import json
import threading
import concurrent.futures


def get_onepage_urls(onepageurl):
    """获取单个翻页的所有图片的urls+当前翻页的下一翻页的url"""
    if not onepageurl:
        print('已到最后一页, 结束')
        return [], ''
    try:
        html = requests.get(onepageurl)
        html.encoding = 'utf-8'
        html = html.text
    except Exception as e:
        print(e)
        pic_urls = []
        fanye_url = ''
        return pic_urls, fanye_url
    pic_urls = re.findall('"objURL":"(.*?)",', html, re.S)
    fanye_urls = re.findall(re.compile(r'<a href="(.*)" class="n">下一页</a>'), html, flags=0)
    fanye_url = 'http://image.baidu.com' + fanye_urls[0] if fanye_urls else ''
    return pic_urls, fanye_url


def down_pic(pic_urls, key_word, save_path):
    """给出图片链接列表, 下载所有图片"""
    for i, pic_url in enumerate(pic_urls):
        try:
            pic = requests.get(pic_url, timeout=15)
            string = save_path + '/' + key_word + '_' + str(i + 1) + '.jpg'
            with open(string, 'wb') as f:
                f.write(pic.content)
                # print('成功下载第%s张图片: %s' % (str(i + 1), str(pic_url)))
        except Exception as e:
            # print('下载第%s张图片时失败: %s' % (str(i + 1), str(pic_url)))
            print(e)
            continue


def download_images_keyword(keyword, page_number=70):
    url_init_first = r'http://image.baidu.com/search/flip?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1497491098685_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&ctd=1497491098685%5E00_1519X735&word='
    url_init = url_init_first + urllib.parse.quote(keyword, safe='/')
    all_pic_urls = []
    onepage_urls, fanye_url = get_onepage_urls(url_init)
    all_pic_urls.extend(onepage_urls)

    fanye_count = 0  # 累计翻页数
    while fanye_count < page_number:
        onepage_urls, fanye_url = get_onepage_urls(fanye_url)
        fanye_count += 1
        if fanye_url == '' and onepage_urls == []:
            break
        all_pic_urls.extend(onepage_urls)

    down_pic(list(set(all_pic_urls)), keyword, save_path)
    

if __name__ == '__main__':
    save_path = '/media/mxq/data/competition/HuaWei/下载的图片/download_images_50pages'
    label_json_path = '/media/mxq/data/competition/HuaWei/label_id_name.json'
    with open(label_json_path, 'r') as f:
        label = json.load(f).values()
    keywords = [keyword.split('/')[1] for keyword in label]
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(download_images_keyword, keywords)

