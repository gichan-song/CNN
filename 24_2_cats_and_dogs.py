# 24_2_cats_and_dogs.py
import os
from pathlib import Path
import shutil
from tensorflow import keras
from keras.api._v2 import keras
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# 퀴즈
# cats_and_dogs 폴더 안쪽에 아래와 같은 폴더를 만드는 함수를 만드세요
# root +-- train +-- cats
#                +-- dogs
#      +-- validation +-- cats
#                     +-- dogs
#      +-- test +-- cats
#               +-- dogs
def make_keras_folders():
    def make_folder_if_not_exist(folder_path):
        # print(Path('cats_and_dogs/root').exists())
        # print(os.path.exists('cats_and_dogs/root'))

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    make_folder_if_not_exist('cats_and_dogs/root/train/cats')
    make_folder_if_not_exist('cats_and_dogs/root/train/dogs')
    make_folder_if_not_exist('cats_and_dogs/root/validation/cats')
    make_folder_if_not_exist('cats_and_dogs/root/validation/dogs')
    make_folder_if_not_exist('cats_and_dogs/root/test/cats')
    make_folder_if_not_exist('cats_and_dogs/root/test/dogs')


def copy_images(src_path, dst_path, start, end):
    kind = 'cat' if dst_path.endswith('cats') else 'dog'

    for i in range(start, end):
        filename = '{}.{}.jpg'.format(kind, i)

        src = os.path.join(src_path, filename)
        dst = os.path.join(dst_path, filename)

        shutil.copy(src, dst)


def generator_basic():
    gen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                       horizontal_flip=True,
                                                       rotation_range=50,
                                                       zoom_range=(0.5, 3.0))
    flow = gen.flow_from_directory('cats_and_dogs/root/train',
                                   target_size=[150, 150],
                                   batch_size=5,
                                   class_mode='binary')   # "categorical", "binary", "sparse"

    # 퀴즈
    # 제너레이터에서 가져온 이미지를 그려보세요
    # for x, y in flow:
    #     print(x.shape, y.shape)       # (5, 150, 150, 3) (5,)
    #     break

    x, y = next(flow)
    print(x.shape, y.shape)             # (5, 150, 150, 3) (5,)

    print(flow.class_indices)           # {'cats': 0, 'dogs': 1}
    # print(flow.filenames[:5])
    # print(flow.index_array)

    plt.figure(figsize=[10, 4])
    for i in range(len(x)):
        plt.subplot(2, len(x), i+1)
        plt.imshow(x[i])
        # plt.title('dog' if y[i] else 'cat')

        idx = flow.index_array[i]
        filename = flow.filenames[idx]
        plt.title(filename)

        img_path = os.path.join('cats_and_dogs/root/train', filename)
        print(img_path)

        img = Image.open(img_path)

        plt.subplot(2, len(x), i+1 + len(x))
        plt.imshow(np.array(img))
        plt.title('original')

    plt.show()


# make_keras_folders()

# copy_images('cats_and_dogs/train', 'cats_and_dogs/root/train/cats', 0, 1000)
# copy_images('cats_and_dogs/train', 'cats_and_dogs/root/train/dogs', 0, 1000)
# copy_images('cats_and_dogs/train', 'cats_and_dogs/root/validation/cats', 1000, 1500)
# copy_images('cats_and_dogs/train', 'cats_and_dogs/root/validation/dogs', 1000, 1500)
# copy_images('cats_and_dogs/train', 'cats_and_dogs/root/test/cats', 1500, 2000)
# copy_images('cats_and_dogs/train', 'cats_and_dogs/root/test/dogs', 1500, 2000)

generator_basic()

