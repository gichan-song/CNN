# AutoEncoder_2.py
from tensorflow import keras
from keras.api._v2 import keras
import matplotlib.pyplot as plt
from sklearn import decomposition


# 이전 파일에서 함수와는 y 데이터를 반환하기 때문에 조금 다릅니다
def get_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    return x_train.reshape(-1, 784), y_train, x_test.reshape(-1, 784), y_test


# values shape: (60000, 2)
def plot_label_clusters(values, labels):
    plt.figure(figsize=[7, 7])
    plt.scatter(values[:, 0], values[:, 1], c=labels, s=3)
    plt.colorbar()
    plt.show()


# x shape: (60000, 28, 28, 1)
def show_pca(x, y):
    pca = decomposition.PCA(n_components=2)
    result = pca.fit_transform(x.reshape(-1, 784))

    plot_label_clusters(result, y)


def make_encoder_1():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer([784]))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(2, activation='relu'))

    return model


def make_decoder_1():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer([2]))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(784, activation='sigmoid'))

    return model


def make_encoder_2():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer([28, 28, 1]))
    model.add(keras.layers.Conv2D(32, [3, 3], 1, 'same'))       # stride: 1
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2D(64, [3, 3], 2, 'same'))       # stride: 2
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2D(64, [3, 3], 2, 'same'))       # stride: 2
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2D(64, [3, 3], 1, 'same'))       # stride: 1
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(2))

    return model


def make_decoder_2():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer([2]))
    model.add(keras.layers.Dense(7 * 7 * 64))
    model.add(keras.layers.Reshape([7, 7, 64]))

    model.add(keras.layers.Conv2DTranspose(64, [3, 3], 1, 'same'))       # stride: 1
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2DTranspose(64, [3, 3], 2, 'same'))       # stride: 2
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2DTranspose(64, [3, 3], 2, 'same'))       # stride: 2
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2DTranspose(32, [3, 3], 1, 'same'))       # stride: 1
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2DTranspose(1, [3, 3], 1, 'same', activation='tanh'))

    return model


def show_model_1():
    x_train, y_train, x_test, y_test = get_mnist()
    # show_pca(x_train, y_train)

    encoder, decoder = make_encoder_1(), make_decoder_1()

    model = keras.Sequential([encoder, decoder])

    model.compile(optimizer=keras.optimizers.Adam(0.0005),
                  loss=keras.losses.binary_crossentropy)

    model.fit(x_train, x_train, epochs=10, batch_size=32, verbose=2)

    p = model.predict(x_train, verbose=0)
    show_pca(p, y_train)


def show_model_2():
    x_train, y_train, x_test, y_test = get_mnist()

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    encoder, decoder = make_encoder_2(), make_decoder_2()
    # encoder.summary()
    # decoder.summary()

    model = keras.Sequential([encoder, decoder])

    model.compile(optimizer=keras.optimizers.Adam(0.0005),
                  loss=keras.losses.binary_crossentropy)

    model_path = 'model/ae_mnist_conv_{epoch:03d}_{val_loss:.4f}.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, period=5)

    model.fit(x_train, x_train, epochs=100, batch_size=32, verbose=1,
              callbacks=[checkpoint],
              validation_data=(x_test, y_test))
    model.save('model/ae_mnist_conv_100.h5')

    p = model.predict(x_train)
    show_pca(p, y_train)


if __name__ == '__main__':
    # show_model_1()
    show_model_2()
