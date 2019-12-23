from autoaugment import ImageNetPolicy
import os
import PIL
import matplotlib.pyplot as plt

dataset_path = '/media/mxq/data/competition/HuaWei/train_data'
images = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
for image in images:
    path = os.path.join(dataset_path, image)
    image = PIL.Image.open(path)
    policy = ImageNetPolicy()
    transformed = policy(image)
    plt.imshow(transformed)
    plt.show()
