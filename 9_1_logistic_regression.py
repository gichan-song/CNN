# 9_1_logistic_regression.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.e ** -z)


def show_sigmoid():
    for z in np.linspace(-10, 10):
        s = sigmoid(z)

        plt.plot(z, s, 'ro')
    plt.show()


def binary_cross_entropy(y, p):
    loss_i = y * -tf.math.log(p) + (1 - y) * -tf.math.log(1 - p)
    return tf.reduce_mean(loss_i)


def logistic_regression():
    def dense(x, w, b):
        return x @ w + b

    x = [[1, 2],        # 탈락
         [2, 1],
         [4, 5],        # 통과
         [5, 4],
         [8, 9],
         [9, 8]]
    y = [[0],
         [0],
         [1],
         [1],
         [1],
         [1]]
    y = np.int32(y)

    w = tf.Variable(tf.random.uniform([2, 1]))
    b = tf.Variable(tf.random.uniform([1]))

    optimizer = tf.keras.optimizers.SGD(0.1)

    for i in range(100):
        with tf.GradientTape() as tape:
            z = dense(x, w, b)
            # hx = sigmoid(z)
            hx = keras.activations.sigmoid(z)

            # loss = binary_cross_entropy(y, hx)
            # bce = keras.losses.BinaryCrossentropy()                   # simple version
            # loss = bce(y, hx)
            loss = keras.losses.binary_crossentropy(y, hx, axis=0)      # full version

        gradient = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(gradient, [w, b]))

        print(i, loss.numpy())
    print()

    z = dense(x, w, b)
    p = sigmoid(z).numpy()
    print(p)

    p_flat = p.reshape(-1)
    print(p_flat)

    p_bool = np.int32(p_flat > 0.5)
    y_flat = y.reshape(-1)
    print(p_bool)
    print(y_flat)
    print()

    equals = (p_bool == y_flat)
    print(equals)
    print('acc :', np.mean(equals))
    print('-' * 30)

    # 퀴즈
    # 3시간 공부하고 7번 출석한 학생과
    # 5시간 공부하고 2번 출석한 학생의 통과 여부를 알려주세요
    z = dense([[3, 7],
               [5, 2]], w, b)
    p = sigmoid(z).numpy()
    print(p)

    p_flat = p.reshape(-1)
    print(p_flat)


def show_plot():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    def CE(x, y):
        return y * (-np.log(x)) + (1 - y) * (-np.log(1 - x))

    def CE2(x, y):
        return (np.log(y)) * (-np.log(x)) + (np.log(1 - y)) * (-np.log(1 - x))

    x = np.linspace(0.0001, 0.9999, 101)
    y = np.linspace(0.0001, 0.9999, 101)
    zz = np.array([[CE(i, j) for j in y] for i in x])
    xx, yy = np.meshgrid(x, y)
    ax.plot_surface(xx, yy, zz)
    ax.set_xlabel('p')
    ax.set_ylabel('y')
    print(np.min(zz), np.max(zz))
    fig.show()
    plt.show()


# show_sigmoid()
# show_plot()

logistic_regression()





