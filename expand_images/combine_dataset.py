import shutil
import os
import tqdm


def combine_dataset(download_root, official_root, combine_root):
    if os.path.exists(combine_root):
        print('Removing %s' % combine_root)
        shutil.rmtree(combine_root)
        print('Making %s' % combine_root)
        os.mkdir(combine_root)
    else:
        print('Making %s' % combine_root)
        os.mkdir(combine_root)        

    download_files = os.listdir(download_root)
    tbar = tqdm.tqdm(download_files)
    for download_file in tbar:
        scr_file = os.path.join(download_root, download_file)
        target_file = os.path.join(combine_root, download_file)
        shutil.copy(scr_file, target_file)
    
    official_files = os.listdir(official_root)
    tbar = tqdm.tqdm(official_files)
    for official_file in tbar:
        scr_file = os.path.join(official_root, official_file)
        target_file = os.path.join(combine_root, official_file)
        shutil.copy(scr_file, target_file)


if __name__ == "__main__":
    download_root = '/media/mxq/data/competition/HuaWei/cleaned_dowload_images'
    official_root = '/media/mxq/data/competition/HuaWei/train_data'
    combine_root = '/media/mxq/data/competition/HuaWei/combine'
    combine_dataset(download_root, official_root, combine_root)
