import os
import shutil

choose_dict = {    
    "19": "景点/西安城墙",
    "20": "景点/钟楼",
    "21": "景点/长安华严寺",
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
    "36": "美食/枸杞炖银耳"
    }

choose_classes = choose_dict.values()
choose_classes = [x.split('/')[-1] for x in choose_classes]

origin_path = './psudeo_image'
save_path = './choose_image'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for name in os.listdir(origin_path):
    if name.split('_')[0] in choose_classes:
        print('copy file {}'.format(name))
        shutil.copy(os.path.join(origin_path, name), save_path)
