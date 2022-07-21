# 11_2_iris_sparse.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing


# 퀴즈
# iris_sparse.csv 파일을 읽어서 품종을 구분하는 딥러닝 모델을 구축하세요 (정확도 표시)
# 70%의 데이터로 학습하고, 30%의 데이터에 대해 정확도를 계산합니다
# iris_sparse.csv 파일의 품종을 0, 1, 2로 변환하는 과정도 포함합니다
def make_xy_3():
    iris = pd.read_csv('data/iris_sparse.csv')
    # print(iris)

    return iris.values[:, :-1], iris.values[:, -1]


# 퀴즈
# LabelEncoder 클래스를 사용해서 sparse 버전의 모델을 구축하세요
def make_xy_sparse():
    iris = pd.read_csv('data/iris.csv')
    # print(iris)

    x = iris.values[:, :-1]
    variety = iris.values[:, -1]

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(variety)
    print(y)

    return x, y


def softmax_regression_iris():
    def dense(x, w, b):
        # (150, 3) = (150, 4) @ (4, 3)
        return x @ w + b

    # x, y = make_xy_3()
    x, y = make_xy_sparse()
    print(x.shape, y.shape)             # (150, 4) (150,)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)

    w = tf.Variable(tf.random.uniform([4, 3]))
    b = tf.Variable(tf.random.uniform([3]))

    optimizer = tf.keras.optimizers.SGD(0.1)

    for i in range(10):
        with tf.GradientTape() as tape:
            z = dense(x_train, w, b)
            hx = keras.activations.softmax(z)

            bcce = keras.losses.SparseCategoricalCrossentropy()
            loss = bcce(y_train, hx)

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
    print(p_arg)

    print('acc :', np.mean(p_arg == y_test))


softmax_regression_iris()
