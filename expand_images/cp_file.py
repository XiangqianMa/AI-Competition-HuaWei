import os
import glob
import shutil
import json
import tqdm


# 将特定文件从一个文件夹拷贝至另一个文件夹
src_root = '/media/mxq/data/competition/HuaWei/choose_image'
target_root = '/media/mxq/data/competition/HuaWei/cleaned_pesudeo_image'
keywords_file = '/media/mxq/data/competition/HuaWei/2.json'
with open(keywords_file, 'r') as f:
    keywords = json.load(f).values()

for keyword in keywords:
    files_name = glob.glob(os.path.join(src_root, '*'+ keyword.split('/')[-1] + '*'))
    tbar = tqdm.tqdm(files_name)
    for file_name in tbar:
        target_name = os.path.join(target_root, file_name.split('/')[-1])
        shutil.copy(file_name, target_name)
        tbar.set_description(desc=file_name)