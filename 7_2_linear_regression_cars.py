# 7_2_linear_regression_cars.py
import tensorflow as tf
import pandas as pd
import matplotlib as plt

# 퀴즈
# cars.csv 파일을 읽어서 x, y를 반환하는 함수를 만드세요
def make_xy_1():
    f = open('data/cars.csv', 'r', encoding='utf-8')

    f.readline()

    x, y = [], []
    for line in f:
        # print(line.strip().split(','))
        _, speed, dist = line.strip().split(',')

        x.append(int(speed))
        y.append(int(dist))

    f.close()
    return x, y


def make_xy_2():
    cars = pd.read_csv('data/cars.csv', index_col=0)
    # print(cars)
    # print()
    #
    # print(cars.speed.values)
    # print(cars['dist'].values)

    return cars.speed, cars.dist


def make_xy_3():
    cars = pd.read_csv('data/cars.csv', index_col=0)
    # print(cars.values)

    x = cars.values[:, 0]
    y = cars.values[:, 1]
    # print(x)
    # print(y)

    return x, y


def linear_regression_cars():
    def predict(x, w, b):
        return w * x + b

    def mean_square_error(y, p):
        return tf.reduce_mean((p - y) ** 2)

    # x, y = make_xy_1()
    # x, y = make_xy_2()
    x, y = make_xy_3()
    # x = [1, 2, 3]       # 공부한 시간
    # y = [1, 2, 3]       # 성적

    w = tf.Variable(tf.random.uniform([1]))
    b = tf.Variable(tf.random.uniform([1]))

    optimizer = tf.keras.optimizers.SGD(0.001)

    for i in range(10):
        with tf.GradientTape() as tape:
            hx = predict(x, w, b)
            loss = mean_square_error(y, hx)

        gradient = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(gradient, [w, b]))

        print(i, loss.numpy())
        plt.plot(i,loss.numpy())
    plt.show()




    # # 퀴즈
    # # 5시간 공부한 학생과
    # # 7시간 공부한 학생의 성적을 구하세요
    # print('* :', predict([5, 7], w, b).numpy())


linear_regression_cars()



