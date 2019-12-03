import os
import shutil
import json
import glob


def clean_data(dataset_root, label_id):
    """ 如果txt名称和txt类标对应的名称不一致则删除该数据

    Args:
        dataset_root: str, 要清洗的数据集路径
        label_id: dict, {名称：label}
    """
    count = 0
    txt_paths = glob.glob(os.path.join(dataset_root, '*.txt'))
    # 处理每一个txt
    for txt_path in txt_paths:
        # 如果不是官方数据集
        if not txt_path.split('/')[-1].startswith('img'):
            # 打开文件
            with open(txt_path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    txt_name = line.split(', ')[0]
                    label_index = int(line.split(', ')[1])
            label_name = label_id[str(label_index)]
            # 如果txt名称和txt类标对应的名称不一致
            if txt_name.split('_')[0] not in label_name:
                if txt_name.split('_')[0] == '浆水鱼鱼' and label_name == '美食/凉鱼':
                    pass
                else:
                    count += 1
                    print(txt_name, label_name)
                    os.remove(txt_path)
                    os.remove(txt_path.replace('.txt', '.jpg'))

    print('Deal {} image'.format(count))


def choose_data(choose_dict):
    """ 用于从数据集中筛选只出现在choose_dict中的样本，并复制到save_path中

    Args:
        choose_dict: dict, {名称：label}
    """
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


if __name__ == '__main__':
    with open("online-service/model/label_id_name.json", 'r', encoding='utf-8') as json_file:
        label_id = json.load(json_file)
    clean_data('data/huawei_data/combine', label_id)
