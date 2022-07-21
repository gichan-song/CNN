# 21_1_cat.py
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.api._v2 import keras


# 퀴즈
# 학습 데이터에서 처음 이미지 5장을 그래프로 출력하세요 (imshow 사용)
def show_cats(images, labels):
    count = len(images)
    plt.figure(figsize=[count * 2, 2])
    for i, img in enumerate(images):
        # print(img.shape)            # (64, 64, 3)
        plt.subplot(1, count, i+1)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.imshow(img)
        plt.title('{}'.format('cat' if labels[i] == 1 else 'non-cat'))

    plt.tight_layout()
    plt.show()


def make_model_by_lenet5():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer([64, 64, 3]))
    model.add(keras.layers.Conv2D(16, [3, 3], 1, 'same'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))
    model.add(keras.layers.Conv2D(32, [3, 3], 1, 'same'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))
    model.add(keras.layers.Conv2D(32, [3, 3], 1, 'same'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))
    model.add(keras.layers.Conv2D(64, [3, 3], 1, 'same'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()
    return model


def make_model_by_vgg():
    model = keras.Sequential([
        keras.layers.InputLayer([64, 64, 3]),

        keras.layers.Conv2D(16, [3, 3], 1, 'same', activation='relu'),
        keras.layers.Conv2D(16, [3, 3], 1, 'same', activation='relu'),
        keras.layers.MaxPool2D([2, 2], 2, 'same'),

        keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'),
        keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'),
        keras.layers.MaxPool2D([2, 2], 2, 'same'),

        keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'),
        keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'),
        keras.layers.MaxPool2D([2, 2], 2, 'same'),

        keras.layers.Conv2D(64, [3, 3], 1, 'same', activation='relu'),
        keras.layers.Conv2D(64, [3, 3], 1, 'same', activation='relu'),
        keras.layers.MaxPool2D([2, 2], 2, 'same'),

        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.summary()
    return model


train_images = h5py.File('data/train_catvnoncat.h5')
test_images = h5py.File('data/test_catvnoncat.h5')
print(type(train_images))
print(train_images.keys())
print(test_images.keys())
# <KeysViewHDF5 ['list_classes', 'train_set_x', 'train_set_y']>
# <KeysViewHDF5 ['list_classes', 'test_set_x', 'test_set_y']>

print(train_images['list_classes'])
# <HDF5 dataset "list_classes": shape (2,), type "|S7">

print(np.array(train_images['list_classes']))   # [b'non-cat' b'cat']

# 퀴즈
# x, y 데이터를 가져와서 shape을 출력하세요
x_train = np.array(train_images['train_set_x'])
y_train = np.array(train_images['train_set_y'])
x_test = np.array(test_images['test_set_x'])
y_test = np.array(test_images['test_set_y'])
print(x_train.shape, y_train.shape)             # (209, 64, 64, 3) (209,)
print(x_test.shape, y_test.shape)               # (50, 64, 64, 3) (50,)

# show_cats(x_train[:7], y_train[:7])
# print(np.min(x_train), np.max(x_train))       # 0 255

x_train = x_train / 255                         # 정규화
x_test = x_test / 255

# 퀴즈
# 전체 사진 중에서 고양이 사진은 몇 장일까요?
print('    cat :', np.sum(y_train))                     # 72
print('non-cat :', len(y_train) - np.sum(y_train))      # 137

# 퀴즈
# 전체 사진 중에서 고양이 사진만 5장 출력하세요
finds = np.where(y_train == 1)
finds = finds[0]
print(finds)                    # [  2   7  11  13  14  19  ...]

# 1번
cats_1 = np.zeros([5, 64, 64, 3], dtype=np.int32)
for i in range(5):
    pos = finds[i]
    # print(x_train[pos].shape)   # (64, 64, 3)
    cats_1[i] = x_train[pos]

# 2번
five = finds[:5]
cats_2 = x_train[five]

# show_cats(cats_1, [1] * len(cats_1))
# show_cats(cats_2, [1] * len(cats_2))

# 퀴즈
# 고양이 데이터셋에 대해 CNN 모델을 구축하고 정확도를 구하세요
# model = make_model_by_lenet5()
model = make_model_by_vgg()

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.binary_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=70, verbose=2,
          validation_data=[x_test, y_test])










