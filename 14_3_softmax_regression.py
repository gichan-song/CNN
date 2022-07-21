# 14_3_softmax_regression.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np


# 퀴즈
# 텐서플로로 구현했던 소프트맥스 리그레션 모델을 케라스 모델로 수정하세요
# 학습 후에 정확도까지 구합니다
def softmax_regression_onehot():
    x = [[1, 2],        # C
         [2, 1],
         [4, 5],        # B
         [5, 4],
         [8, 9],        # A
         [9, 8]]
    y = [[0, 0, 1],     # one-hot vector
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]

    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=10, verbose=2)
    print('acc :', model.evaluate(x, y, verbose=0))

    p = model.predict(x, verbose=0)
    print(p)

    p_arg = np.argmax(p, axis=1)
    y_arg = np.argmax(y, axis=1)
    print(p_arg)
    print(y_arg)

    print('acc :', np.mean(p_arg == y_arg))


# 퀴즈
# 원핫 벡터 버전을 스파스 버전으로 수정하세요
def softmax_regression_sparse():
    x = [[1, 2],        # C
         [2, 1],
         [4, 5],        # B
         [5, 4],
         [8, 9],        # A
         [9, 8]]
    y = [2, 2, 1, 1, 0, 0]

    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=1000, verbose=2)
    print('acc :', model.evaluate(x, y, verbose=0))

    p = model.predict(x, verbose=0)
    print(p)

    p_arg = np.argmax(p, axis=1)
    print(p_arg)

    print('acc :', np.mean(p_arg == y))


# softmax_regression_onehot()
softmax_regression_sparse()
