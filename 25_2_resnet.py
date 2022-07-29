# 25_2_resnet.py
from tensorflow import keras
from keras.api._v2 import keras


# 퀴즈
# 레즈넷에서 얘기하는 plain34 모델을 만드세요 (summary로 확인만)
def plain_34():
    inputs = keras.layers.Input([224, 224, 3])
    x = keras.layers.Conv2D(64, [7, 7], 2, 'same')(inputs)  # 1/2
    x = keras.layers.MaxPool2D([2, 2], 2, 'same')(x)        # 1/2

    x = keras.layers.Conv2D(64, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(64, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(64, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(64, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(64, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(64, [3, 3], 1, 'same')(x)

    x = keras.layers.Conv2D(128, [3, 3], 2, 'same')(x)      # 1/2
    x = keras.layers.Conv2D(128, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(128, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(128, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(128, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(128, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(128, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(128, [3, 3], 1, 'same')(x)

    x = keras.layers.Conv2D(256, [3, 3], 2, 'same')(x)  # 1/2
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)

    x = keras.layers.Conv2D(512, [3, 3], 2, 'same')(x)  # 1/2
    x = keras.layers.Conv2D(512, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(512, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(512, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(512, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(512, [3, 3], 1, 'same')(x)

    # x = keras.layers.Flatten()(x)                     # wrong
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(1000, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.summary()


# 퀴즈
# 앞에서 만든 34개 레이어 모델을 bottle neck 구조를 추가해서 50개 레이어 모델로 수정하세요
def plain_50():
    inputs = keras.layers.Input([224, 224, 3])
    x = keras.layers.Conv2D(64, [7, 7], 2, 'same')(inputs)  # 1/2
    x = keras.layers.MaxPool2D([2, 2], 2, 'same')(x)        # 1/2

    x = keras.layers.Conv2D(64, [1, 1], 1, 'same')(x)
    x = keras.layers.Conv2D(64, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [1, 1], 1, 'same')(x)



    x = keras.layers.Conv2D(64, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(64, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(64, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(64, [3, 3], 1, 'same')(x)

    x = keras.layers.Conv2D(128, [3, 3], 2, 'same')(x)      # 1/2
    x = keras.layers.Conv2D(128, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(128, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(128, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(128, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(128, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(128, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(128, [3, 3], 1, 'same')(x)

    x = keras.layers.Conv2D(256, [3, 3], 2, 'same')(x)  # 1/2
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(256, [3, 3], 1, 'same')(x)

    x = keras.layers.Conv2D(512, [3, 3], 2, 'same')(x)  # 1/2
    x = keras.layers.Conv2D(512, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(512, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(512, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(512, [3, 3], 1, 'same')(x)
    x = keras.layers.Conv2D(512, [3, 3], 1, 'same')(x)

    # x = keras.layers.Flatten()(x)                     # wrong
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(1000, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.summary()


# plain_34()
plain_50()
