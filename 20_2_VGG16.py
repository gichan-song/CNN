# 20_2_VGG16.py
from tensorflow import keras
from keras.api._v2 import keras


# 퀴즈
# VGG16 모델을 만드세요
# 파라미터 갯수가 1억 3천 8백만개쯤인지 확인합니다
def vgg16_normal():
    model = keras.Sequential([
        keras.layers.InputLayer([224, 224, 3]),

        keras.layers.Conv2D(64, [3, 3], 1, 'same', activation='relu'),
        keras.layers.Conv2D(64, [3, 3], 1, 'same', activation='relu'),
        keras.layers.MaxPool2D([2, 2], 2, 'same'),

        keras.layers.Conv2D(128, [3, 3], 1, 'same', activation='relu'),
        keras.layers.Conv2D(128, [3, 3], 1, 'same', activation='relu'),
        keras.layers.MaxPool2D([2, 2], 2, 'same'),

        keras.layers.Conv2D(256, [3, 3], 1, 'same', activation='relu'),
        keras.layers.Conv2D(256, [3, 3], 1, 'same', activation='relu'),
        keras.layers.Conv2D(256, [3, 3], 1, 'same', activation='relu'),
        keras.layers.MaxPool2D([2, 2], 2, 'same'),

        keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'),
        keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'),
        keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'),
        keras.layers.MaxPool2D([2, 2], 2, 'same'),

        keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'),
        keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'),
        keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'),
        keras.layers.MaxPool2D([2, 2], 2, 'same'),

        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dense(1000, activation='softmax'),
    ])

    model.summary()


# 퀴즈
# 앞에서 만든 vgg16_normal 함수를 반복문 버전으로 수정하세요
def vgg16_loop_1():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer([224, 224, 3]))

    for _ in range(2):
        model.add(keras.layers.Conv2D(64, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))

    for _ in range(2):
        model.add(keras.layers.Conv2D(128, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))

    for _ in range(3):
        model.add(keras.layers.Conv2D(256, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))

    for _ in range(3):
        model.add(keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))

    for _ in range(3):
        model.add(keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(1000, activation='softmax'))

    model.summary()


def vgg16_loop_2(layers):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer([224, 224, 3]))

    for _ in range(layers[0]):
        model.add(keras.layers.Conv2D(64, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))

    for _ in range(layers[1]):
        model.add(keras.layers.Conv2D(128, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))

    for _ in range(layers[2]):
        model.add(keras.layers.Conv2D(256, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))

    for _ in range(layers[3]):
        model.add(keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))

    for _ in range(layers[4]):
        model.add(keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(1000, activation='softmax'))

    model.summary()


def vgg16_loop_3(layers):
    def conv_block(model, n_loop, n_layer):
        for _ in range(n_loop):
            model.add(keras.layers.Conv2D(n_layer, [3, 3], 1, 'same', activation='relu'))
        model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))

    model = keras.Sequential()
    model.add(keras.layers.InputLayer([224, 224, 3]))

    # conv_block(model, layers[0], 64)
    # conv_block(model, layers[1], 128)
    # conv_block(model, layers[2], 256)
    # conv_block(model, layers[3], 512)
    # conv_block(model, layers[4], 512)

    for n_loop, n_layer in layers:
        conv_block(model, n_loop, n_layer)

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(1000, activation='softmax'))

    model.summary()


# vgg16_normal()
# vgg16_loop_1()
# vgg16_loop_2([2, 2, 3, 3, 3])
# vgg16_loop_2([2, 2, 4, 4, 4])
vgg16_loop_3([(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)])
