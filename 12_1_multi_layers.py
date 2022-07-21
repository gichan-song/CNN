# 12_1_multi_layers.py
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


# xavier
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


# 퀴즈
# mnist 데이터셋에 대해서 소프트맥스 리그레션 모델을 사용해서 정확도를 구하세요
def mnist_single_layer():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)
    print(y_train.shape, y_test.shape)      # (60000,) (10000,)

    print(y_train[:5])                      # [5 0 4 1 9]

    x_train = x_train.reshape(-1, 784)      # (60000, 784)
    x_test = x_test.reshape(-1, 784)        # (10000, 784)

    print(np.min(x_train), np.max(x_train)) # 0 255

    x_train = x_train / 255                 # 정규화
    x_test = x_test / 255

    n_classes = 10
    # w, b = make_weight_normal(x_train.shape[-1], n_classes)
    # w, b = make_weight_glorot(x_train.shape[-1], n_classes)
    w, b = make_weight_he(x_train.shape[-1], n_classes)

    # optimizer = keras.optimizers.SGD(0.1)
    # optimizer = keras.optimizers.RMSprop(0.01)
    optimizer = keras.optimizers.Adam(0.01)

    for i in range(100):
        with tf.GradientTape() as tape:
            hx = dense(x_train, w, b, activation=keras.activations.softmax)

            bcce = keras.losses.SparseCategoricalCrossentropy()
            loss = bcce(y_train, hx)

        gradient = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(gradient, [w, b]))

        print(i, loss.numpy())
    print()

    p = dense(x_test, w, b, activation=keras.activations.softmax)
    print(p.numpy().shape)

    p_arg = np.argmax(p.numpy(), axis=1)
    print('acc :', np.mean(p_arg == y_test))


def mnist_single_layer_mini_batch():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 784)      # (60000, 784)
    x_test = x_test.reshape(-1, 784)        # (10000, 784)

    x_train = x_train / 255                 # 정규화
    x_test = x_test / 255

    n_classes = 10
    # w, b = make_weight_normal(x_train.shape[-1], n_classes)
    w, b = make_weight_glorot(x_train.shape[-1], n_classes)
    # w, b = make_weight_he(x_train.shape[-1], n_classes)

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
                hx = dense(xx, w, b, activation=keras.activations.softmax)

                bcce = keras.losses.SparseCategoricalCrossentropy()
                loss = bcce(yy, hx)
                total += loss.numpy()

            gradient = tape.gradient(loss, [w, b])
            optimizer.apply_gradients(zip(gradient, [w, b]))

        print(i, total / n_iteration)
    print()

    p = dense(x_test, w, b, activation=keras.activations.softmax)
    print(p.numpy().shape)

    p_arg = np.argmax(p.numpy(), axis=1)
    print('acc :', np.mean(p_arg == y_test))


# mnist_single_layer()
mnist_single_layer_mini_batch()








