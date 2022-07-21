# 8_1_multiple_regression.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
import numpy as np


def mean_square_error(y, p):
    return tf.reduce_mean((p - y) ** 2)


def multiple_regression_1():
    def predict(x1, x2, w1, w2, b):
        return w1 * x1 + w2 * x2 + b

    # hx = w1 * x1 + w2 * x2 + b
    #       1         1        0
    # y  =      x1 +      x2
    x1 = [1, 2, 4, 5, 8, 9]         # 공부한 시간
    x2 = [2, 1, 5, 4, 9, 8]         # 출석한 일수
    y = [3, 3, 9, 9, 17, 17]        # 성적

    w1 = tf.Variable(tf.random.uniform([1]))
    w2 = tf.Variable(tf.random.uniform([1]))
    b = tf.Variable(tf.random.uniform([1]))

    optimizer = tf.keras.optimizers.SGD(0.01)

    for i in range(10):
        with tf.GradientTape() as tape:
            hx = predict(x1, x2, w1, w2, b)
            loss = mean_square_error(y, hx)

        gradient = tape.gradient(loss, [w1, w2, b])
        optimizer.apply_gradients(zip(gradient, [w1, w2, b]))

        print(i, loss.numpy())

    # 퀴즈
    # 3시간 공부하고 7번 출석한 학생과
    # 5시간 공부하고 2번 출석한 학생의 성적을 구하세요
    print(predict(x1, x2, w1, w2, b).numpy())
    print(predict([1, 2, 4, 5, 8, 9], [2, 1, 5, 4, 9, 8], w1, w2, b).numpy())
    print(predict([3, 5], [7, 2], w1, w2, b).numpy())


# 퀴즈
# 여러 개로 나누어진 피처 변수 x1과 x2를 하나의 변수로 통합하세요
def multiple_regression_2():
    def predict(x, w, b):
        return w[0] * x[0] + w[1] * x[1] + b

    x = [[1, 2, 4, 5, 8, 9],        # 공부한 시간
         [2, 1, 5, 4, 9, 8]]        # 출석한 일수
    y = [3, 3, 9, 9, 17, 17]        # 성적

    w = tf.Variable(tf.random.uniform([2]))
    b = tf.Variable(tf.random.uniform([1]))

    optimizer = tf.keras.optimizers.SGD(0.01)

    for i in range(10):
        with tf.GradientTape() as tape:
            hx = predict(x, w, b)
            loss = mean_square_error(y, hx)

        gradient = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(gradient, [w, b]))

        print(i, loss.numpy())

    # 퀴즈
    # 3시간 공부하고 7번 출석한 학생과
    # 5시간 공부하고 2번 출석한 학생의 성적을 구하세요
    print(predict([[3, 5],
                   [7, 2]], w, b).numpy())


# 퀴즈
# bias를 없애보세요
# [1 2 3] + 1 = [2 3 4]                         # broadcast
# [1 2 3] + [1 1 1] = [2 3 4]                   # vector operation
def multiple_regression_3():
    # 1: w[0] * x[0] + w[1] * x[1] + b
    # 2: w[0] * x[0] + w[1] * x[1] + w[2]
    # 3: w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
    # 4: w[0] * x[0] + w[1] * x[1] + w[2] * [1, 1, 1, 1, 1, 1]
    def predict(x, w):
        return w[0] * x[0] + w[1] * x[1] + w[2] * x[2]

    # x = [[1, 2, 4, 5, 8, 9],        # 공부한 시간
    #      [2, 1, 5, 4, 9, 8],        # 출석한 일수
    #      [1, 1, 1, 1, 1, 1]]        # bias
    x = [[1, 1, 1, 1, 1, 1],        # bias
         [1, 2, 4, 5, 8, 9],        # 공부한 시간
         [2, 1, 5, 4, 9, 8]]        # 출석한 일수
    y = [3, 3, 9, 9, 17, 17]        # 성적

    w = tf.Variable(tf.random.uniform([3]))

    optimizer = tf.keras.optimizers.SGD(0.01)

    for i in range(1000):
        with tf.GradientTape() as tape:
            hx = predict(x, w)
            loss = mean_square_error(y, hx)

        gradient = tape.gradient(loss, [w])
        optimizer.apply_gradients(zip(gradient, [w]))

        if i % 10 == 0:
            print(i, loss.numpy())

    # 퀴즈
    # 3시간 공부하고 7번 출석한 학생과
    # 5시간 공부하고 2번 출석한 학생의 성적을 구하세요
    # print(predict([[3, 5],
    #                [7, 2],
    #                [1, 1]], w).numpy())
    print(predict([[1, 1],
                   [3, 5],
                   [7, 2]], w).numpy())
    print()
    print(w.numpy())


# 퀴즈
# 앞에서 만든 predict 함수를 행렬 곱셈으로 수정하세요
def multiple_regression_4():
    # (2, 3) = (2, 5) @ (5, 3)
    # (1, 6) = (1, 3) @ (3, 6)
    def predict(x, w):
        return w @ x

    x = [[1, 1, 1, 1, 1, 1],        # bias
         [1, 2, 4, 5, 8, 9],        # 공부한 시간
         [2, 1, 5, 4, 9, 8]]        # 출석한 일수
    y = [3, 3, 9, 9, 17, 17]        # 성적

    w = tf.Variable(tf.random.uniform([1, 3]))

    optimizer = tf.keras.optimizers.SGD(0.01)

    for i in range(10):
        with tf.GradientTape() as tape:
            hx = predict(x, w)
            loss = mean_square_error(y, hx)

        gradient = tape.gradient(loss, [w])
        optimizer.apply_gradients(zip(gradient, [w]))

        print(i, loss.numpy())

    # 퀴즈
    # 3시간 공부하고 7번 출석한 학생과
    # 5시간 공부하고 2번 출석한 학생의 성적을 구하세요
    print(predict([[1, 1],
                   [3, 5],
                   [7, 2]], w).numpy())


# 퀴즈
# x, y 데이터를 정상적인 형태로 수정하세요
def multiple_regression_5():
    # (6, 1) = (6, 3) @ (3, 1)
    def predict(x, w):
        return x @ w

    x = [[1, 1, 2],
         [1, 2, 1],
         [1, 4, 5],
         [1, 5, 4],
         [1, 8, 9],
         [1, 9, 8]]
    y = [[3],
         [3],
         [9],
         [9],
         [17],
         [17]]

    w = tf.Variable(tf.random.uniform([3, 1]))

    optimizer = tf.keras.optimizers.SGD(0.01)

    for i in range(10):
        with tf.GradientTape() as tape:
            hx = predict(x, w)
            loss = mean_square_error(y, hx)

        gradient = tape.gradient(loss, [w])
        optimizer.apply_gradients(zip(gradient, [w]))

        print(i, loss.numpy())

    # 퀴즈
    # 3시간 공부하고 7번 출석한 학생과
    # 5시간 공부하고 2번 출석한 학생의 성적을 구하세요
    print(predict([[1, 3, 7],
                   [1, 5, 2]], w).numpy())


# 퀴즈
# x 데이터에서 bias를 제거하세요
def multiple_regression_6():
    # (6, 1) = (6, 2) @ (2, 1)
    def dense(x, w, b):
        # return x @ w + b
        return tf.matmul(x, w) + b

    x = [[1, 2],
         [2, 1],
         [4, 5],
         [5, 4],
         [8, 9],
         [9, 8]]
    y = [[3],
         [3],
         [9],
         [9],
         [17],
         [17]]

    x = np.float32(x)

    w = tf.Variable(tf.random.uniform([2, 1]))
    b = tf.Variable(tf.random.uniform([1]))

    optimizer = tf.keras.optimizers.SGD(0.01)

    for i in range(10):
        with tf.GradientTape() as tape:
            hx = dense(x, w, b)

            # loss = mean_square_error(y, hx)
            mse = keras.losses.MeanSquaredError()
            loss = mse.__call__(y, hx)

        gradient = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(gradient, [w, b]))

        print(i, loss.numpy())

    # 퀴즈
    # 3시간 공부하고 7번 출석한 학생과
    # 5시간 공부하고 2번 출석한 학생의 성적을 구하세요
    print(dense([[3., 7.],
                 [5., 2.]], w, b).numpy())


# multiple_regression_1()
# multiple_regression_2()
# multiple_regression_3()
# multiple_regression_4()
# multiple_regression_5()
multiple_regression_6()

# a = np.arange(6).reshape(-1, 1)     # (6, 1)
# b = np.arange(6)                    # (1, 6) <- (6,)
# print(a)
# print(b)
# print()
#
# print(a - b)                        # (6, 6) = (6, 1) - (1, 6)








