# 22_1_VGG16_1x1.py
from tensorflow import keras
from keras.api._v2 import keras


# 퀴즈
# 20-2 파일에서 만든 VGG16 모델에서
# 덴스 레이어를 컨볼루션 레이어로 대체하세요
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

    # keras.layers.Flatten(),
    # keras.layers.Dense(4096, activation='relu'),
    # keras.layers.Dense(4096, activation='relu'),
    # keras.layers.Dense(1000, activation='softmax'),

    keras.layers.Conv2D(4096, [7, 7], 1, 'valid', activation='relu'),   # valid 중요
    keras.layers.Conv2D(4096, [1, 1], 1, 'same', activation='relu'),
    keras.layers.Conv2D(1000, [1, 1], 1, 'same'),
    keras.layers.Flatten(),
    keras.layers.Softmax(),
])

model.summary()

