# 18_1_functional.py
from tensorflow import keras
from keras.api._v2 import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# 퀴즈
# 아래 데이터(AND)에 대해 동작하는 케라스 모델을 구축하세요
def and_sequential():
    data = [[1, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]]
    data = np.int32(data)

    x = data[:, :-1]        # (4, 2)
    y = data[:, -1:]        # (4, 1)

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[2]))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=100, verbose=2)


def xor_sequential():
    data = [[1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 0]]
    data = np.int32(data)

    x = data[:, :-1]        # (4, 2)
    y = data[:, -1:]        # (4, 1)

    tf.random.set_seed(2)

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[2]))
    model.add(keras.layers.Dense(2, activation='sigmoid'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.Adam(0.05),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=120, verbose=2)


def and_functional():
    data = [[1, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]]
    data = np.int32(data)

    x = data[:, :-1]
    y = data[:, -1:]

    # ------------------------------------------------ #

    # model = keras.Sequential([
    #     keras.layers.InputLayer(input_shape=[2]),
    #     keras.layers.Dense(1, activation='sigmoid'),
    # ])

    inputs = keras.layers.Input(shape=[2])
    dense = keras.layers.Dense(1, activation='sigmoid')

    # model = keras.Sequential([inputs, dense])

    # output = dense.__call__(inputs)
    # output = dense(inputs)

    output = keras.layers.Dense(1, activation='sigmoid')(inputs)

    model = keras.Model(inputs, output)
    # model.summary()

    # ------------------------------------------------ #

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=100, verbose=2)


# 퀴즈
# xor_sequential 함수를 펑셔널 모델로 수정하세요
def xor_functional_1():
    data = [[1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 0]]
    data = np.int32(data)

    x = data[:, :-1]        # (4, 2)
    y = data[:, -1:]        # (4, 1)

    tf.random.set_seed(2)

    inputs = keras.layers.Input(shape=[2])
    output = keras.layers.Dense(2, activation='sigmoid')(inputs)
    output = keras.layers.Dense(1, activation='sigmoid')(output)

    model = keras.Model(inputs, output)

    model.compile(optimizer=keras.optimizers.Adam(0.05),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=120, verbose=2)


def xor_functional_2():
    data = [[1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 0]]
    data = np.int32(data)

    x1 = data[:, :1]        # (4, 1)
    x2 = data[:, 1:2]       # (4, 1)
    y = data[:, -1:]        # (4, 1)

    tf.random.set_seed(2)

    inputs_1 = keras.layers.Input(shape=[1])
    output_1 = keras.layers.Dense(1, activation='sigmoid')(inputs_1)

    inputs_2 = keras.layers.Input(shape=[1])
    output_2 = keras.layers.Dense(1, activation='sigmoid')(inputs_2)

    output = keras.layers.concatenate([output_1, output_2], axis=1)
    output = keras.layers.Dense(1, activation='sigmoid')(output)

    model = keras.Model([inputs_1, inputs_2], output)

    model.compile(optimizer=keras.optimizers.Adam(0.05),
                  loss=keras.losses.mse,                # 엉뚱한 손실 함수에도 동작
                  metrics=['acc'])

    model.fit([x1, x2], y, epochs=120, verbose=2)


# 퀴즈
# 앞에서 만든 모델에 AND 데이터를 추가한 모델을 구축하세요
def xor_functional_3():
    data = [[1, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]]
    data = np.int32(data)

    x = data[:, :2]         # (4, 2)
    y1 = data[:, -2:-1]     # (4, 1)
    y2 = data[:, -1:]       # (4, 1)

    tf.random.set_seed(2)

    inputs = keras.layers.Input(shape=[2])
    output = keras.layers.Dense(2, activation='sigmoid')(inputs)

    output_xor = keras.layers.Dense(1, activation='sigmoid', name='xor')(output)
    output_and = keras.layers.Dense(1, activation='sigmoid', name='and')(output)

    model = keras.Model(inputs, [output_xor, output_and])
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.05),
                  loss=[keras.losses.mse, keras.losses.binary_crossentropy],
                  metrics=['acc'])

    history = model.fit(x, [y1, y2], epochs=120, verbose=2)
    print(history.history.keys())
    # dict_keys(['loss', 'xor_loss', 'and_loss', 'xor_acc', 'and_acc'])

    # 퀴즈
    # 위의 모델에서 반환한 결과를 시각화 하세요
    xor_loss = history.history['xor_loss']
    and_loss = history.history['and_loss']

    xor_acc = history.history['xor_acc']
    and_acc = history.history['and_acc']

    x = np.arange(len(xor_loss))

    plt.subplot(1, 2, 1)
    plt.plot(x, xor_loss, label='xor')
    plt.plot(x, and_loss, label='and')
    plt.title('loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, xor_acc, label='xor')
    plt.plot(x, and_acc, label='and')
    plt.title('acc')
    plt.legend()

    plt.show()


# and_sequential()
# xor_sequential()

# and_functional()
# xor_functional_2()
xor_functional_3()
