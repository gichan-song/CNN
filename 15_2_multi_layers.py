# 15_2_multi_layers.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용


# 퀴즈
# 아래 코드가 반환하는 mnist 데이터셋에 대해 정확도를 계산하는 싱글 레이어 모델을 만드세요
def mnist_layer_single():
    data = keras.datasets.mnist.load_data()
    print(type(data), len(data))
    print(type(data[0]))

    (x_train, y_train), (x_test, y_test) = data
    print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
    print(y_train.shape, y_test.shape)  # (60000,) (10000,)

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    x_train = x_train / 255  # 정규화
    x_test = x_test / 255

    model = keras.Sequential()
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2)
    print('acc :', model.evaluate(x_test, y_test, verbose=0))


# 퀴즈
# 앞에서 만든 싱글 레이어 버전을 멀티 레이버 버전으로 수정하세요
def mnist_layer_multiple():
    data = keras.datasets.mnist.load_data()

    (x_train, y_train), (x_test, y_test) = data

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    x_train = x_train / 255  # 정규화
    x_test = x_test / 255

    # 784 -> 256 -> 256 -> 10
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=[784]))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2)
    print('acc :', model.evaluate(x_test, y_test, verbose=0))


# mnist_layer_single()
mnist_layer_multiple()
