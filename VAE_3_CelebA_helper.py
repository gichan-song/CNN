# VAE_3_CelebA_helper.py
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import os


def get_image_paths(image_folder, n_train, n_valid, n_test, seed=None):
    if seed:
        np.random.seed(seed)

    filenames = list(os.listdir(image_folder))
    image_paths = [os.path.join(image_folder, name) for name in filenames]
    np.random.shuffle(image_paths)
    # print(image_paths[:3])

    p1 = n_train
    p2 = p1 + n_valid
    p3 = p2 + n_test

    return image_paths[:p1], image_paths[p1:p2], image_paths[p2:p3]


def save_images_to_npy(image_paths, npy_path, size):
    images = []
    for p in image_paths:
        img = Image.open(p)
        img = img.resize((size, size))
        images.append(np.array(img))

    np.save(npy_path, images)


# 퀴즈
# 전달된 파일로부터 앞에 나오는 이미지 5장을 그래프로 출력하세요
def show_images_from_npy(npy_path):
    images = np.load(npy_path)
    print(images.shape)         # (1000, 64, 64, 3)

    plt.figure(figsize=[10, 2])
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i])

    plt.show()


# 이미지 경로 목록으로부터 5장을 출력
def show_images_from_image_paths(image_paths):
    plt.figure(figsize=[10, 2])
    for i in range(5):
        plt.subplot(1, 5, i+1)
        img = Image.open(image_paths[i])
        plt.imshow(img)

    plt.show()


# 퀴즈
# 아래 속성을 매개 변수로 받아서
# 해당 속성을 갖는 이미지 파일 이름만 반환하는 함수를 만드세요
# 5_o_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes, Bald,Bangs,
# Big_Lips,Big_Nose,Black_Hair,Blond_Hair,Blurry,
# Brown_Hair,Bushy_Eyebrows,Chubby,Double_Chin,Eyeglasses,
# Goatee,Gray_Hair,Heavy_Makeup,High_Cheekbones,Male,
# Mouth_Slightly_Open,Mustache,Narrow_Eyes,No_Beard,Oval_Face,
# Pale_Skin,Pointy_Nose,Receding_Hairline,Rosy_Cheeks,Sideburns,
# Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,Wearing_Hat,
# Wearing_Lipstick,Wearing_Necklace,Wearing_Necktie,Young
def get_attr_images(attr):
    attr_celeb = pd.read_csv('celeb_a/list_attr_celeba.csv', index_col=0)
    # print(attr_celeb)

    wanted = attr_celeb[attr]
    # print(wanted)

    found = (wanted == 1)
    # print(found)

    return wanted[found]


def show_attr_images(attr):
    filenames = get_attr_images(attr)
    # print(filenames.index.values)         # ['000109.jpg' '000209.jpg' ... '202543.jpg' '202588.jpg']
    # print(filenames.index.values.dtype)   # object

    image_paths = [os.path.join('celeb_a/img_align_celeba', name) for name in filenames.index.values]
    show_images_from_image_paths(image_paths)


# data = get_image_paths('celeb_a/img_align_celeba',
#                        n_train=1000, n_valid=1000, n_test=1000, seed=23)
# train_paths, valid_paths, test_paths = data
#
# save_images_to_npy(train_paths, 'celeb_a/train_1000.npy', size=64)
# save_images_to_npy(valid_paths, 'celeb_a/valid_1000.npy', size=64)
# save_images_to_npy(test_paths, 'celeb_a/test_1000.npy', size=64)

# show_images_from_npy('celeb_a/train_1000.npy')

# show_attr_images('Mustache')
# show_attr_images('Smiling')
