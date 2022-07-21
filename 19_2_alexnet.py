# 19_2_alexnet.py
from tensorflow import keras
from keras.api._v2 import keras


# 퀴즈
# 교재에서 봤던 알렉스넷을 아키텍처만 구현하세요
inputs = keras.layers.Input([224, 224, 3])
output = keras.layers.Conv2D(96, [11, 11], 4, 'same', activation='relu')(inputs)
output = keras.layers.MaxPool2D([3, 3], 2, 'valid')(output)
output = keras.layers.Conv2D(256, [5, 5], 1, 'same', activation='relu')(output)
output = keras.layers.MaxPool2D([3, 3], 2, 'valid')(output)
output = keras.layers.Conv2D(384, [3, 3], 1, 'same', activation='relu')(output)
output = keras.layers.Conv2D(384, [3, 3], 1, 'same', activation='relu')(output)
output = keras.layers.Conv2D(256, [3, 3], 1, 'same', activation='relu')(output)
output = keras.layers.MaxPool2D([3, 3], 2, 'valid')(output)
output = keras.layers.Flatten()(output)
output = keras.layers.Dense(4096, activation='relu')(output)
output = keras.layers.Dense(4096, activation='relu')(output)
output = keras.layers.Dense(1000, activation='softmax')(output)

model = keras.Model(inputs, output)
model.summary()
