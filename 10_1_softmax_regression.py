# 10_1_softmax_regression.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
import numpy as np
import matplotlib.pyplot as plt


# onehot = np.identity(6)
# print(onehot)
#
# y_sparse = [2, 1, 0, 3, 4]
# y = onehot[y_sparse]
# print(y)


def softmax(z):
    # np.sum을 사용하면 미분이 정확하게 되지 않을 수도 있습니다
    s = tf.exp(z)
    t = tf.reduce_sum(s, axis=1)
    # print(t.numpy())                  # [6. 9.]

    # (2, 3) = (2, 3) / (2, 1)          # vector + broadcast
    return s / tf.reshape(t, [-1, 1])


# z = [[1., 2., 3.],
#      [2., 4., 3.]]
# print(softmax(z).numpy())               # [[0.16666667 0.33333334 0.5       ]
#                                         #  [0.22222222 0.44444445 0.33333334]]


# 퀴즈
# 소프트맥스에서 사용하는 손실 함수를 직접 만들어 보세요
def categorical_cross_entropy(y, p):
    loss_i = tf.reduce_sum(y * -tf.math.log(p), axis=1)
    return tf.reduce_mean(loss_i)


# 퀴즈
# sparse 버전의 손실 함수를 만드세요
def sparse_categorical_cross_entropy(y_sparse, p):
    onehot = np.identity(p.shape[-1])
    y = onehot[y_sparse]

    return categorical_cross_entropy(y, p)


def softmax_regression():
    def dense(x, w, b):
        # (6, 3) = (6, 2) @ (2, 3)
        return x @ w + b

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
    y = np.int32(y)

    w = tf.Variable(tf.random.uniform([2, 3]))
    b = tf.Variable(tf.random.uniform([3]))

    optimizer = tf.keras.optimizers.SGD(0.01)

    for i in range(100):
        with tf.GradientTape() as tape:
            z = dense(x, w, b)
            hx = softmax(z)
            # hx = keras.activations.softmax(z)

            loss = categorical_cross_entropy(y, hx)
            # cce = keras.losses.CategoricalCrossentropy()
            # loss = cce(y, hx)

        gradient = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(gradient, [w, b]))

        print(i, loss.numpy())
    print()

    z = dense(x, w, b)
    p = softmax(z)
    print(p.numpy())
    print(p.numpy().shape)              # (6, 3)

    # 퀴즈
    # 3시간 공부하고 7번 출석한 학생과
    # 5시간 공부하고 2번 출석한 학생의 학점을 알려주세요
    result = np.argmax(p.numpy(), axis=1)
    print(result)

    grades = np.array(['A', 'B', 'C'])
    print(grades[result])


def softmax_regression_sparse():
    def dense(x, w, b):
        # (6, 3) = (6, 2) @ (2, 3)
        return x @ w + b

    x = [[1, 2],        # C
         [2, 1],
         [4, 5],        # B
         [5, 4],
         [8, 9],        # A
         [9, 8]]
    y = [2, 2, 1, 1, 0, 0]
    y = np.int32(y)

    w = tf.Variable(tf.random.uniform([2, 3]))
    b = tf.Variable(tf.random.uniform([3]))

    optimizer = tf.keras.optimizers.SGD(0.01)

    for i in range(100):
        with tf.GradientTape() as tape:
            z = dense(x, w, b)
            hx = softmax(z)
            # hx = keras.activations.softmax(z)

            loss = sparse_categorical_cross_entropy(y, hx)
            # scce = keras.losses.SparseCategoricalCrossentropy()
            # loss = scce(y, hx)

        gradient = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(gradient, [w, b]))

        print(i, loss.numpy())
    print()

    z = dense(x, w, b)
    p = softmax(z)
    print(p.numpy())
    print(p.numpy().shape)              # (6, 3)

    # 퀴즈
    # 3시간 공부하고 7번 출석한 학생과
    # 5시간 공부하고 2번 출석한 학생의 학점을 알려주세요
    result = np.argmax(p.numpy(), axis=1)
    print(result)

    grades = np.array(['A', 'B', 'C'])
    print(grades[result])


# softmax_regression()
softmax_regression_sparse()





