# 14_2_logistic_regression.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np


# 퀴즈
# 텐서플로로 구현했던 로지스틱 리그레션 모델을 케라스 모델로 수정하세요
# 학습 후에 정확도까지 구합니다
def logistic_regression():
    x = [[1, 2],
         [2, 1],
         [4, 5],
         [5, 4],
         [8, 9],
         [9, 8]]
    y = [[0],
         [0],
         [1],
         [1],
         [1],
         [1]]

    model = keras.Sequential()
    model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))

    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=10, verbose=2)
    print('acc :', model.evaluate(x, y, verbose=0))

    p = model.predict(x, verbose=0)
    print(p)

    p_flat = np.int32(p > 0.5).reshape(-1)
    y_flat = np.reshape(y, -1)
    print(p_flat)
    print(y_flat)

    print('acc :', np.mean(p_flat == y_flat))

    # 퀴즈
    # 3시간 공부하고 7번 출석한 학생과
    # 5시간 공부하고 2번 출석한 학생의 통과 여부를 구하세요
    print(model.predict([[3, 7],
                         [5, 2]], verbose=0))


# 퀴즈
# pima-indians-diabetes.csv 파일을 읽어서 당뇨병을 판단하는 딥러닝 모델을 구축하세요 (정확도 표시)
# 70%의 데이터로 학습하고, 30%의 데이터에 대해 정확도를 계산합니다
def logistic_regression_indian():
    indian = pd.read_csv('data/pima-indians-diabetes.csv', skiprows=9, header=None)

    x = indian.values[:, :-1]
    y = indian.values[:, -1:]
    print(x.shape, y.shape)         # (768, 8) (768, 1)

    x = preprocessing.scale(x)
    # x = preprocessing.minmax_scale(x)

    data = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['acc'])

    # model.fit(x_train, y_train, epochs=1000, batch_size=len(x_train), verbose=2)
    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2)
    print('acc :', model.evaluate(x_test, y_test, verbose=0))

    p = model.predict(x_test, verbose=0)

    p_flat = np.int32(p > 0.5).reshape(-1)
    y_flat = np.reshape(y_test, -1)

    print('acc :', np.mean(p_flat == y_flat))


# logistic_regression()
logistic_regression_indian()
