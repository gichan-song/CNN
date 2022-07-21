# 14_1_multiple_regression.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
import pandas as pd
from sklearn import preprocessing


# 퀴즈
# 텐서플로로 만들었던 멀티플 리그레션을 케라스 버전으로 수정하세요
def multiple_regression():
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

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.mse)

    model.fit(x, y, epochs=10, verbose=2)
    print('mse :', model.evaluate(x, y, verbose=0))

    print(model.predict(x, verbose=0))

    # 퀴즈
    # 3시간 공부하고 7번 출석한 학생과
    # 5시간 공부하고 2번 출석한 학생의 성적을 구하세요
    print(model.predict([[3, 7],
                         [5, 2]], verbose=0))


# 퀴즈
# trees.csv 파일에 대해 텐서플로를 사용해서 구축했던 모델을 케라스 버전으로 다시 만드세요
def multiple_regression_trees_wrong():
    trees = pd.read_csv('data/trees.csv', index_col=0)
    # print(trees)

    x = trees.values[:, :-1]
    y = trees.values[:, -1:]

    # scale 함수를 2회 호출하면 안된다는 것을 증명하는 코드
    # print(x[:2])
    #
    # z = preprocessing.scale(x)
    # print(z[:2])
    # print(preprocessing.scale(x[:2]))

    # 퀴즈
    # 기존 코드에 스케일링을 적용해 보세요
    x = preprocessing.scale(x)              # loss: 36.9140
    # x = preprocessing.minmax_scale(x)     # loss: 131.0729

    print(x.shape, y.shape)     # (31, 2) (31, 1)

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.mse)

    model.fit(x, y, epochs=10, verbose=2)

    # 퀴즈
    # Girth가 10이고 Height가 80인 나무와
    # Girth가 15이고 Height가 90인 나무의 Volume을 예측하세요
    z = preprocessing.scale([[10, 80],
                             [15, 90]])         # 틀린 코드
    p = model.predict(z)
    print(p)


def multiple_regression_trees_right():
    trees = pd.read_csv('data/trees.csv', index_col=0)
    # print(trees)

    x = trees.values[:, :-1]
    y = trees.values[:, -1:]

    # 2회에 걸쳐 스케일링을 진행할 때 사용하는 올바른 코드
    # scaler = preprocessing.StandardScaler()
    # z = scaler.fit_transform(x)
    #
    # print(x[:2])
    # print(z[:2])
    # print(scaler.transform(x[:2]))

    scaler = preprocessing.StandardScaler()
    x = scaler.fit_transform(x)

    print(x.shape, y.shape)     # (31, 2) (31, 1)

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.mse)

    model.fit(x, y, epochs=10, verbose=2)

    # 퀴즈
    # Girth가 10이고 Height가 80인 나무와
    # Girth가 15이고 Height가 90인 나무의 Volume을 예측하세요
    z = scaler.transform([[10, 80],
                          [15, 90]])         # 맞는 코드
    p = model.predict(z)
    print(p)


# multiple_regression()
# multiple_regression_trees_wrong()
multiple_regression_trees_right()
