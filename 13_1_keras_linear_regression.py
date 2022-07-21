# 13_1_keras_linear_regression.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
import pandas as pd


def linear_regression():
    x = [[1],
         [2],
         [3]]       # 공부한 시간
    y = [[1],
         [2],
         [3]]       # 성적

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.mse)

    model.fit(x, y, epochs=1000, verbose=2)   # verbose(0: 출력 없음, 1: 전체 출력, 2: 일부 출력)
    print('mse :', model.evaluate(x, y, verbose=0))

    print(model.predict(x, verbose=0))

    # # # 퀴즈
    # # # 5시간 공부한 학생과
    # # # 7시간 공부한 학생의 성적을 구하세요
    print(model.predict([[5],
                         [7]], verbose=0))


# 퀴즈
# cars.csv 파일에 대해 텐서플로를 사용해서 구축했던 모델을 케라스 버전으로 다시 만드세요
def linear_regression_cars():
    cars = pd.read_csv('data/cars.csv', index_col=0)

    # x = cars.speed.values.reshape(-1, 1)
    # y = cars.dist.values.reshape(-1, 1)

    x = cars.values[:, :1]
    y = cars.values[:, 1:]

    print(x.shape, y.shape)     # (50, 1) (50, 1)

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.SGD(0.001),
                  loss=keras.losses.mse)

    model.fit(x, y, epochs=10, verbose=2)
    print('mse :', model.evaluate(x, y, verbose=0))

    p = model.predict([[0], [30]])
    print(p)


# linear_regression()
linear_regression_cars()
