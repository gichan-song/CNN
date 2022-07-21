# 16_1_boston_housing_price.py
# http://192.168.0.11/CNN/
# pandas, scikit-learn, matplotlib, tensorflow
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt


# 프로젝트
# BostonHousing.xls 파일을 읽어서
# 80%의 데이터로 학습하고 20%의 데이터에 대해 mae를 계산하세요
# mae: mean absolute error
# y를 medv 컬럼 사용
def model_boston_multiple_regression():
    boston = pd.read_excel('data/BostonHousing.xls')
    print(boston)

    x = boston.values[:, :-2]
    y = boston.values[:, -2:-1]
    print(x.shape, y.shape)  # (506, 13) (506, 1)

    x = preprocessing.scale(x)
    # x = preprocessing.minmax_scale(x)

    data = model_selection.train_test_split(x, y, train_size=0.8)
    x_train, x_test, y_train, y_test = data

    # 13 -> 8 -> 4 -> 1
    model = keras.Sequential()
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(4, activation='relu'))
    model.add(keras.layers.Dense(1))
    # model.add(keras.layers.Dense(32, activation='relu'))
    # model.add(keras.layers.Dense(16, activation='relu'))
    # model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss=keras.losses.mse,
                  metrics=['mae', 'mape'])      # keras.metrics.mape

    history = model.fit(x_train, y_train, epochs=10,
                        verbose=2,
                        validation_data=[x_test, y_test])
    print('mae :', model.evaluate(x_test, y_test, verbose=0))

    # 퀴즈
    # mae를 predict 함수 반환값에 적용해 보세요
    p = model.predict(x_test, verbose=0)
    error = np.absolute(p - y_test)
    print('mae :', np.mean(error))

    # 퀴즈
    # mape를 predict 함수 반환값에 적용해 보세요
    # 평균(abs(예측결과 - 정답) / 정답) * 100
    print('mape :', np.mean(error / y_test) * 100)

    print(type(history.history))
    print(history.history.keys())
    # dict_keys(['loss', 'mae', 'val_loss', 'val_mae'])

    print(history.history['loss'])
    print(len(history.history['loss']))

    # 퀴즈
    # history에 포함된 데이터 중에서 손실 그래프, mae 그래프를 그려보세요
    x = range(len(history.history['loss']))

    plt.subplot(1, 2, 1)
    plt.plot(x, history.history['loss'], label='loss')
    plt.plot(x, history.history['val_loss'], label='val_loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, history.history['mae'], label='mae')
    plt.plot(x, history.history['val_mae'], label='val_mae')
    plt.legend()

    plt.show()


# 프로젝트
# BostonHousing.xls 파일을 읽어서
# 80%의 데이터로 학습하고 20%의 데이터에 대해 정확도를 계산하세요
# y를 "cat. medv" 컬럼 사용
def model_boston_logistic_regression():
    boston = pd.read_excel('data/BostonHousing.xls')
    # print(boston)
    # print(boston['CAT. MEDV'])
    # print(set(boston['CAT. MEDV']))         # {0, 1}

    x = boston.values[:, :-2]
    y = boston.values[:, -1:]
    print(x.shape, y.shape)  # (506, 13) (506, 1)

    x = preprocessing.scale(x)
    # x = preprocessing.minmax_scale(x)

    data = model_selection.train_test_split(x, y, train_size=0.8)
    x_train, x_test, y_train, y_test = data

    # 13 -> 8 -> 4 -> 1
    model = keras.Sequential()
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(4, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['acc'])

    history = model.fit(x_train, y_train, epochs=10,
                        verbose=2,
                        validation_data=[x_test, y_test])
    # print('mae :', model.evaluate(x_test, y_test, verbose=0))

    # 퀴즈
    # mae를 predict 함수 반환값에 적용해 보세요
    # p = model.predict(x_test, verbose=0)
    # print('mae :', np.mean(np.absolute(p - y_test)))

    print(type(history.history))
    print(history.history.keys())
    # dict_keys(['loss', 'mae', 'val_loss', 'val_mae'])

    print(history.history['loss'])
    print(len(history.history['loss']))

    # 퀴즈
    # history에 포함된 데이터 중에서 손실 그래프, mae 그래프를 그려보세요
    x = range(len(history.history['loss']))

    plt.subplot(1, 2, 1)
    plt.plot(x, history.history['loss'], label='loss')
    plt.plot(x, history.history['val_loss'], label='val_loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, history.history['acc'], label='acc')
    plt.plot(x, history.history['val_acc'], label='val_acc')
    plt.legend()

    plt.tight_layout()
    plt.show()


model_boston_multiple_regression()
# model_boston_logistic_regression()
