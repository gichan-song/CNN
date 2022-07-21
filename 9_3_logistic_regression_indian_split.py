# 9_3_logistic_regression_indian_split.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
import numpy as np
import pandas as pd
from sklearn import preprocessing


# 퀴즈
# pima-indians-diabetes.csv 파일을 읽어서 당뇨병을 판단하는 딥러닝 모델을 구축하세요 (정확도 표시)
# 70%의 데이터로 학습하고, 30%의 데이터에 대해 정확도를 계산합니다
def make_xy_3():
    indian = pd.read_csv('data/diabetes.csv', skiprows=9, header=None)
    # print(indian)

    # 예측 결과와 비교하기 쉽도록 int32로 변환
    return indian.values[:, :-1], np.int32(indian.values[:, -1:])


def logistic_regression_indian():
    def dense(x, w, b):
        # (768, 1) = (768, 8) @ (8, 1)
        # (600, 1) = (600, 8) @ (8, 1)
        # (168, 1) = (168, 8) @ (8, 1)
        return x @ w + b

    x, y = make_xy_3()
    # print(x.shape, y.shape)               # (768, 8) (768, 1)

    # x = preprocessing.scale(x)              # 표준화. acc : 0.7965367965367965
    x = preprocessing.minmax_scale(x)       # 정규화

    # train_size = 600
    train_size = int(len(x) * 0.7)

    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print(x_train.shape, x_test.shape)      # (537, 8) (231, 8)
    print(y_train.shape, y_test.shape)      # (537, 1) (231, 1)

    w = tf.Variable(tf.random.uniform([8, 1]))
    b = tf.Variable(tf.random.uniform([1]))

    optimizer = tf.keras.optimizers.SGD(0.01)

    for i in range(1000):
        with tf.GradientTape() as tape:
            z = dense(x_train, w, b)
            hx = keras.activations.sigmoid(z)

            bce = keras.losses.BinaryCrossentropy()
            loss = bce(y_train, hx)

        gradient = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(gradient, [w, b]))

        if i % 10 == 0:
            print(i, loss.numpy())
    print()

    z = dense(x_test, w, b)
    p = keras.activations.sigmoid(z).numpy()
    # print(p)

    p_flat = (p > 0.5).astype(np.int32).reshape(-1)
    y_flat = y_test.reshape(-1)
    print(p_flat[:10])
    print(y_flat[:10])

    print('acc :', np.mean(p_flat == y_flat))


logistic_regression_indian()













