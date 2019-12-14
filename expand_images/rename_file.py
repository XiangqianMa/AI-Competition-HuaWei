import os

path = '/media/mxq/data/competition/HuaWei/combine'
files = os.listdir(path)
image_files = [file for file in files if file.endswith('jpg')]
for image_file in image_files:
    print(image_file)
    txt_file = image_file.replace('jpg', 'txt')
    rename_image_file = image_file.replace('_', '_new_')
    txt_file_path = os.path.join(path, txt_file)
    with open(txt_file_path, 'r') as f:
        for line in f.readlines():
            label = line.split(', ')[1]
    new_line = rename_image_file + ', ' + label
    with open(txt_file_path, 'w') as f:
        f.writelines(new_line)
    image_file_path = os.path.join(path, image_file)
    rename_image_file_path = os.path.join(path, rename_image_file)
    os.rename(image_file_path, rename_image_file_path)
    rename_txt_file_path = os.path.join(path, rename_image_file.replace('jpg', 'txt'))
    os.rename(txt_file_path, rename_txt_file_path)
