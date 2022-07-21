# 12_2_multi_layers.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
import numpy as np


def dense(x, w, b, activation=None):
    return activation(x @ w + b) if activation else (x @ w + b)


def make_weight_normal(n_input, n_output):
    # (60000, 10) = (60000, 784) @ (784, 10)
    # w = tf.Variable(tf.random.uniform([784, 10]))
    # b = tf.Variable(tf.random.uniform([10]))

    w = tf.Variable(tf.random.uniform([n_input, n_output]))
    b = tf.Variable(tf.random.uniform([n_output]))

    return w, b


def make_weight_glorot(n_input, n_output):
    glorot = keras.initializers.GlorotUniform()
    w = tf.Variable(glorot([n_input, n_output]))
    b = tf.Variable(tf.zeros([n_output]))

    return w, b


def make_weight_he(n_input, n_output):
    he = keras.initializers.HeUniform()
    w = tf.Variable(he([n_input, n_output]))
    b = tf.Variable(tf.zeros([n_output]))

    return w, b


def mnist_multiple_layer_mini_batch_3():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 784)      # (60000, 784)
    x_test = x_test.reshape(-1, 784)        # (10000, 784)

    x_train = x_train / 255                 # 정규화
    x_test = x_test / 255

    n_classes = 10

    w1, b1 = make_weight_normal(x_train.shape[-1], 256)
    w2, b2 = make_weight_normal(256, 256)
    w3, b3 = make_weight_normal(256, n_classes)
    # 8 0.15596788449832577
    # 9 0.17600843695608395
    # acc : 0.9573

    # w1, b1 = make_weight_glorot(x_train.shape[-1], 256)
    # w2, b2 = make_weight_glorot(256, 256)
    # w3, b3 = make_weight_glorot(256, n_classes)
    # 8 0.08092645953870184
    # 9 0.0725066219555932
    # acc : 0.9674

    # w1, b1 = make_weight_he(x_train.shape[-1], 256)
    # w2, b2 = make_weight_he(256, 256)
    # w3, b3 = make_weight_he(256, n_classes)
    # 8 0.08378144203420865
    # 9 0.08379319465188019
    # acc : 0.9707

    optimizer = keras.optimizers.Adam(0.01)

    epoch = 10
    batch_size = 100
    n_iteration = len(x_train) // batch_size        # 600

    for i in range(epoch):
        total = 0
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size

            xx = x_train[n1:n2]
            yy = y_train[n1:n2]

            with tf.GradientTape() as tape:
                d1 = dense(xx, w1, b1, activation=keras.activations.relu)
                d2 = dense(d1, w2, b2, activation=keras.activations.relu)
                hx = dense(d2, w3, b3, activation=keras.activations.softmax)

                bcce = keras.losses.SparseCategoricalCrossentropy()
                loss = bcce(yy, hx)
                total += loss.numpy()

            gradient = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
            optimizer.apply_gradients(zip(gradient, [w1, b1, w2, b2, w3, b3]))

        print(i, total / n_iteration)
    print()

    # 퀴즈
    # 정확도를 구해보세요
    d1 = dense(x_test, w1, b1, activation=keras.activations.relu)
    d2 = dense(d1, w2, b2, activation=keras.activations.relu)
    p = dense(d2, w3, b3, activation=keras.activations.softmax)
    print(p.numpy().shape)

    p_arg = np.argmax(p.numpy(), axis=1)
    print('acc :', np.mean(p_arg == y_test))


# 퀴즈
# 레이어 3개짜리를 5개짜리로 늘려서 정확도가 어떻게 달라지는지 확인하세요
def mnist_multiple_layer_mini_batch_5():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 784)      # (60000, 784)
    x_test = x_test.reshape(-1, 784)        # (10000, 784)

    x_train = x_train / 255                 # 정규화
    x_test = x_test / 255

    n_classes = 10

    w1, b1 = make_weight_glorot(x_train.shape[-1], 256)
    w2, b2 = make_weight_glorot(256, 256)
    w3, b3 = make_weight_glorot(256, 128)
    w4, b4 = make_weight_glorot(128, 128)
    w5, b5 = make_weight_glorot(128, n_classes)
    # batch_size: 100, 셔플 안함
    # 8 0.023205870457985233
    # 9 0.01994098744822016
    # acc : 0.9786

    # batch_size: 32, 셔플 안함
    # 8 0.02615837517009454
    # 9 0.027114650519846933
    # acc : 0.9767

    # batch_size: 100, 셔플함
    # 8 0.021656911313257296
    # 9 0.017974101498196737
    # acc : 0.9749

    optimizer = keras.optimizers.Adam(0.001)

    epoch = 10
    batch_size = 100
    n_iteration = len(x_train) // batch_size

    indices = np.arange(len(x_train))

    for i in range(epoch):
        np.random.shuffle(indices)

        x_train = x_train[indices]
        y_train = y_train[indices]

        total = 0
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size

            xx = x_train[n1:n2]
            yy = y_train[n1:n2]

            with tf.GradientTape() as tape:
                d1 = dense(xx, w1, b1, activation=keras.activations.relu)
                d2 = dense(d1, w2, b2, activation=keras.activations.relu)
                d3 = dense(d2, w3, b3, activation=keras.activations.relu)
                d4 = dense(d3, w4, b4, activation=keras.activations.relu)
                hx = dense(d4, w5, b5, activation=keras.activations.softmax)

                bcce = keras.losses.SparseCategoricalCrossentropy()
                loss = bcce(yy, hx)
                total += loss.numpy()

            gradient = tape.gradient(loss, [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5])
            optimizer.apply_gradients(zip(gradient, [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5]))

        print(i, total / n_iteration)
    print()

    # 퀴즈
    # 정확도를 구해보세요
    d1 = dense(x_test, w1, b1, activation=keras.activations.relu)
    d2 = dense(d1, w2, b2, activation=keras.activations.relu)
    d3 = dense(d2, w3, b3, activation=keras.activations.relu)
    d4 = dense(d3, w4, b4, activation=keras.activations.relu)
    p = dense(d4, w5, b5, activation=keras.activations.softmax)
    print(p.numpy().shape)

    p_arg = np.argmax(p.numpy(), axis=1)
    print('acc :', np.mean(p_arg == y_test))


# mnist_multiple_layer_mini_batch_3()
mnist_multiple_layer_mini_batch_5()
