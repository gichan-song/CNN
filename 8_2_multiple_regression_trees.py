# 8_2_multiple_regression_trees.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
import numpy as np
import pandas as pd


# 퀴즈
# trees.csv 파일을 읽어서 리그레션 모델을 구축하고
# Girth가 10이고 Height가 80인 나무와
# Girth가 15이고 Height가 90인 나무의 Volume을 예측하세요
def make_xy_3():
    trees = pd.read_csv('data/trees.csv', index_col=0)
    # print(trees)
    # print(trees.values)
    # print(trees.values.shape)       # (31, 3)

    # x = trees.values[:, :2]
    # y = trees.values[:, 2:]
    x = trees.values[:, :-1]
    y = trees.values[:, -1:]
    # print(x.shape)                  # (31, 2)
    # print(y.shape)                  # (31, 1)

    return np.float32(x), y


def multiple_regression_trees():
    def dense(x, w, b):
        # return x @ w + b
        return tf.matmul(x, w) + b

    x, y = make_xy_3()
    # x = [[1, 2],
    #      [2, 1],
    #      [4, 5],
    #      [5, 4],
    #      [8, 9],
    #      [9, 8]]
    # y = [[3],
    #      [3],
    #      [9],
    #      [9],
    #      [17],
    #      [17]]
    #
    # x = np.float32(x)

    w = tf.Variable(tf.random.uniform([2, 1]))
    b = tf.Variable(tf.random.uniform([1]))

    optimizer = tf.keras.optimizers.SGD(0.0001)

    for i in range(10):
        with tf.GradientTape() as tape:
            hx = dense(x, w, b)

            mse = keras.losses.MeanSquaredError()
            loss = mse(y, hx)

        gradient = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(gradient, [w, b]))

        print(i, loss.numpy())

    print(dense([[10., 70.],
                 [15., 80.]], w, b).numpy())


multiple_regression_trees()





