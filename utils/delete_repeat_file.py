# coding=utf-8

import os
import os.path
import hashlib


def get_md5(filename):
    m = hashlib.md5()
    mfile = open(filename, "rb")
    m.update(mfile.read())
    mfile.close()
    md5_value = m.hexdigest()
    return md5_value


if __name__ == '__main__':
    ipath = "/media/mxq/data/competition/HuaWei/combine"

    filenames = [file for file in os.listdir(ipath) if file.endswith('jpg')]
    md5_dir = {}
    count = 0
    for filename in filenames:
        current_md5 = get_md5(os.path.join(ipath, filename))
        if current_md5 in md5_dir.keys():
            md5_dir[current_md5].append(filename)
            count += 1
            os.remove(os.path.join(ipath, filename))
        else:
            md5_dir[current_md5] = [filename]
    for key, values in md5_dir.items():
        if len(values) != 1:
            print(key + ":", values)

    # 删除单个的txt文件
    filenames = os.listdir(ipath)
    txt_filenames = [file for file in os.listdir(ipath) if file.endswith('txt')]
    for txt_filename in txt_filenames:
        if txt_filename.replace('txt', 'jpg') not in filenames:
            print(txt_filename)
            os.remove(os.path.join(ipath, txt_filename))
