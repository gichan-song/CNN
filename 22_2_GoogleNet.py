# 22_2_GoogleNet.py

# 원본
# * * * * *
# * * * * *
# * * * * *
# * * * * *
# * * * * *

# 목표
# 5x5 영역에서 숫자 1개를 만드는 것

# 필터 (패딩 valid, 스트라이드 1)
# 2 x 2 = 4 x 4
#         2 x 2 = 3 x 3
#                 2 x 2 = 2 x 2
#                         2 x 2 = 1     = 16

# 3 x 3 = 3 x 3
#         3 x 3 = 1                     = 18

# 1 x 5 = 5 x 1
#         5 x 1 = 1                     = 10

# 5 x 5 = 1                             = 25

from tensorflow import keras
from keras.api._v2 import keras


def inception_c_part():
    inputs = keras.layers.Input([8, 8, 1536])
    x = keras.layers.Conv2D(384, [1, 1], 1, 'same', name='1x1')(inputs)

    x1 = keras.layers.Conv2D(256, [1, 3], 1, 'same', name='1x3')(x)
    x2 = keras.layers.Conv2D(256, [3, 1], 1, 'same', name='3x1')(x)

    outputs = keras.layers.concatenate([x1, x2], axis=3, name='concat')

    model = keras.Model(inputs, outputs)
    model.summary()


# inception_c_part()

