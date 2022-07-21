# 9_2_logistic_regression_indian.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
import numpy as np
import pandas as pd

# 퀴즈
# pima-indians-diabetes.csv 파일을 읽어서 당뇨병을 판단하는 딥러닝 모델을 구축하세요 (정확도 표시)
def make_xy_3():
    indian = pd.read_csv('data/pima-indians-diabetes.csv', skiprows=9, header=None)
    # print(indian)

    # 예측 결과와 비교하기 쉽도록 int32로 변환
    return indian.values[:, :-1], np.int32(indian.values[:, -1:])


def logistic_regression_indian():
    def dense(x, w, b):
        # (768, 8) @ (8, 1)
        return x @ w + b

    x, y = make_xy_3()
    # print(x.shape, y.shape)       # (768, 8) (768, 1)

    # x = [[1, 2],        # 탈락
    #      [2, 1],
    #      [4, 5],        # 통과
    #      [5, 4],
    #      [8, 9],
    #      [9, 8]]
    # y = [[0],
    #      [0],
    #      [1],
    #      [1],
    #      [1],
    #      [1]]
    # y = np.int32(y)

    w = tf.Variable(tf.random.uniform([8, 1]))
    b = tf.Variable(tf.random.uniform([1]))

    optimizer = tf.keras.optimizers.SGD(0.001)

    for i in range(100):
        with tf.GradientTape() as tape:
            z = dense(x, w, b)
            hx = keras.activations.sigmoid(z)

            bce = keras.losses.BinaryCrossentropy()                   # simple version
            loss = bce(y, hx)

        gradient = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(gradient, [w, b]))

        print(i, loss.numpy())
    print()

    z = dense(x, w, b)
    p = keras.activations.sigmoid(z).numpy()
    # print(p)

    p_flat = (p > 0.5).astype(np.int32).reshape(-1)
    y_flat = y.reshape(-1)
    print(p_flat[:10])
    print(y_flat[:10])

    print('acc :', np.mean(p_flat == y_flat))


logistic_regression_indian()










