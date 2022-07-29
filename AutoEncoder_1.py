# AutoEncoder_1.py
from tensorflow import keras
from keras.api._v2 import keras
import matplotlib.pyplot as plt


def get_mnist():
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    return x_train.reshape(-1, 784), x_test.reshape(-1, 784)


def show_generation(samples, p):
    plt.figure(figsize=[20, 4])

    for i in range(10):
        ax = plt.subplot(2, 10, i+1)
        plt.imshow(samples[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, 10, i+1+10)
        plt.imshow(p[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()


# 퀴즈
# 인코더와 디코더에 레이어가 1개인 모델을 만드세요
def make_model_single():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer([784]))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(784, activation='sigmoid'))

    return model


# 퀴즈
# 싱글 레이어 모델을 멀티 레이어 버전으로 수정하고 결과를 확인하세요
def make_model_multi():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer([784]))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(784, activation='sigmoid'))

    return model


def make_model_convolution():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer([28, 28, 1]))
    model.add(keras.layers.Conv2D(16, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2))
    model.add(keras.layers.Conv2D(8, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2))
    # model.add(keras.layers.Conv2D(8, [3, 3], 1, 'same', activation='relu'))
    # model.add(keras.layers.MaxPool2D([2, 2], 2))

    # model.add(keras.layers.Conv2D(8, [3, 3], 1, 'same', activation='relu'))
    # model.add(keras.layers.UpSampling2D([2, 2]))
    model.add(keras.layers.Conv2D(8, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.UpSampling2D([2, 2]))
    model.add(keras.layers.Conv2D(16, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.UpSampling2D([2, 2]))
    model.add(keras.layers.Conv2D(1, [3, 3], 1, 'same', activation='sigmoid'))

    return model


# model = make_model_single()
# model = make_model_multi()
model = make_model_convolution()
model.summary()

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.binary_crossentropy)

x_train, x_test = get_mnist()
# print(x_train.shape, x_test.shape)

x_train = x_train.reshape(-1, 28, 28, 1)        # 컨볼루션에서만 사용
x_test = x_test.reshape(-1, 28, 28, 1)

model.fit(x_train, x_train, epochs=10, batch_size=128, verbose=2)

# 퀴즈
# 테스트 데이터 앞쪽에 있는 10개 이미지에 대해 새로운 이미지를 생성하고
# 생성한 이미지를 그래프에 그려주세요
samples = x_test[:10]

p = model.predict(samples, verbose=0)
# print(p.shape)                        # (10, 784) 또는 (10, 28, 28, 1)

show_generation(samples, p)

