# 11_1_iris.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing


def softmax(z):
    # np.sum을 사용하면 미분이 정확하게 되지 않을 수도 있습니다
    s = tf.exp(z)
    t = tf.reduce_sum(s, axis=1)
    # print(t.numpy())                  # [6. 9.]

    # (2, 3) = (2, 3) / (2, 1)          # vector + broadcast
    return s / tf.reshape(t, [-1, 1])


def categorical_cross_entropy(y, p):
    loss_i = tf.reduce_sum(y * -tf.math.log(p), axis=1)
    return tf.reduce_mean(loss_i)


# 퀴즈
# iris_onehot.csv 파일을 읽어서 품종을 구분하는 딥러닝 모델을 구축하세요 (정확도 표시)
# 70%의 데이터로 학습하고, 30%의 데이터에 대해 정확도를 계산합니다
def make_xy_3():
    iris = pd.read_csv('data/iris_onehot.csv')
    # print(iris)

    return iris.values[:, :-3], iris.values[:, -3:]


def make_xy_onehot_1():
    iris = pd.read_csv('data/iris.csv')
    # print(iris)

    x = iris.values[:, :-1]
    variety = iris.values[:, -1]
    print(x)
    print(variety)

    bin = preprocessing.LabelBinarizer()
    y = bin.fit_transform(variety)
    print(y)

    return x, y


def make_xy_onehot_2():
    iris = pd.read_csv('data/iris.csv')
    # print(iris)

    variety = iris.variety
    print(variety)

    bin = preprocessing.LabelBinarizer()
    y = bin.fit_transform(variety)
    print(y)

    df = iris.drop(['variety'], axis=1)
    print(df)

    x = df.values
    return x, y


def softmax_regression_iris():
    def dense(x, w, b):
        # (150, 3) = (150, 4) @ (4, 3)
        return x @ w + b

    # x, y = make_xy_3()
    # x, y = make_xy_onehot_1()
    x, y = make_xy_onehot_2()
    print(x.shape, y.shape)             # (150, 4) (150, 3)
    # x = [[1, 2],        # C
    #      [2, 1],
    #      [4, 5],        # B
    #      [5, 4],
    #      [8, 9],        # A
    #      [9, 8]]
    # y = [[0, 0, 1],     # one-hot vector
    #      [0, 0, 1],
    #      [0, 1, 0],
    #      [0, 1, 0],
    #      [1, 0, 0],
    #      [1, 0, 0]]
    # y = np.int32(y)

    # x, y를 싱크를 맞춘 상태로 셔플
    # indices = np.arange(len(x))
    # np.random.shuffle(indices)
    #
    # x = x[indices]
    # y = y[indices]
    #
    # train_size = int(len(x) * 0.7)
    #
    # x_train, x_test = x[:train_size], x[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]

    # x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)                 # 75:25
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)   # 7:3

    w = tf.Variable(tf.random.uniform([4, 3]))      # 3(setosa, versicolor, virginica)
    b = tf.Variable(tf.random.uniform([3]))

    optimizer = tf.keras.optimizers.SGD(0.1)

    for i in range(1000):
        with tf.GradientTape() as tape:
            z = dense(x_train, w, b)
            hx = softmax(z)
            # hx = keras.activations.softmax(z)

            # cce = keras.losses.CategoricalCrossentropy()
            # loss = cce(y_train, hx)
            loss = categorical_cross_entropy(y_train, hx)

        gradient = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(gradient, [w, b]))

        print(i, loss.numpy())
    print()

    z = dense(x_test, w, b)
    p = keras.activations.softmax(z)
    print(p.numpy())
    print(p.numpy().shape)              # (45, 3)

    print(y_test[:5])
    print(p.numpy()[:5])
    print()

    p_arg = np.argmax(p.numpy(), axis=1)
    y_arg = np.argmax(y_test, axis=1)
    print(p_arg)
    print(y_arg)

    print('acc :', np.mean(p_arg == y_arg))


softmax_regression_iris()
