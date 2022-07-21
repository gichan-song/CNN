# 15_1_softmax_regression_iris.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np


# 퀴즈
# iris.csv 파일을 읽어서 품종을 구분하는 원핫 버전의 딥러닝 모델을 구축하세요 (정확도 표시)
# 70%의 데이터로 학습하고, 30%의 데이터에 대해 정확도를 계산합니다
def softmax_regression_iris_onehot():
    iris = pd.read_csv('data/iris.csv')
    # print(iris)

    x = iris.values[:, :-1]
    print(x.shape, x.dtype)         # (150, 4) object

    x = np.float32(x)               # object -> float32

    # label to binary
    bin = preprocessing.LabelBinarizer()
    y = bin.fit_transform(iris.variety)
    print(y[:10])
    print(y.shape)                  # (150, 3)

    data = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=1000, batch_size=len(x_train), verbose=2)
    print('acc :', model.evaluate(x_test, y_test, verbose=0))

    p = model.predict(x_test, verbose=0)

    p_arg = np.argmax(p, axis=1)
    y_arg = np.argmax(y_test, axis=1)

    print('acc :', np.mean(p_arg == y_arg))


# 퀴즈
# iris.csv 파일을 읽어서 품종을 구분하는 sparse 버전의 딥러닝 모델을 구축하세요 (정확도 표시)
# 70%의 데이터로 학습하고, 30%의 데이터에 대해 정확도를 계산합니다
def softmax_regression_iris_sparse():
    iris = pd.read_csv('data/iris.csv')
    # print(iris)

    x = iris.values[:, :-1]
    print(x.shape, x.dtype)         # (150, 4) object

    x = np.float32(x)               # object -> float32

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(iris.variety)
    print(y[:10])                   # [0 0 0 0 0 0 0 0 0 0]
    print(y.shape)                  # (150,)

    data = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=1000, verbose=2)
    print('acc :', model.evaluate(x_test, y_test, verbose=0))

    p = model.predict(x_test, verbose=0)
    p_arg = np.argmax(p, axis=1)

    print('acc :', np.mean(p_arg == y_test))


# softmax_regression_iris_onehot()
softmax_regression_iris_sparse()
