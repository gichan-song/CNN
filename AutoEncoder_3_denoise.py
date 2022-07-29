# AutoEncoder_3_denoise.py
from tensorflow import keras
from keras.api._v2 import keras
import numpy as np
import matplotlib.pyplot as plt


def get_mnist():
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    return x_train.reshape(-1, 784), x_test.reshape(-1, 784)


def show_images(samples):
    plt.figure(figsize=[20, 2])

    for i in range(10):
        ax = plt.subplot(1, 10, i+1)
        plt.imshow(samples[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    # plt.show()


def make_encoder():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer([28, 28, 1]))
    model.add(keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))
    model.add(keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))

    return model


def make_decoder():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer([7, 7, 32]))
    model.add(keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.UpSampling2D([2, 2]))
    model.add(keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.UpSampling2D([2, 2]))
    model.add(keras.layers.Conv2D(1, [3, 3], 1, 'same', activation='sigmoid'))

    return model


def denoise_auto_encoder():
    x_train, x_test = get_mnist()

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    encoder, decoder = make_encoder(), make_decoder()
    # encoder.summary()
    # decoder.summary()

    noise_factor = 0.5
    x_train += noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test  += noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noise = np.clip(x_train, 0, 1)
    x_test_noise = np.clip(x_test, 0, 1)

    show_images(x_test_noise[:10])

    model = keras.Sequential([encoder, decoder])

    model.compile(optimizer=keras.optimizers.Adam(0.0005),
                  loss=keras.losses.binary_crossentropy)

    model.fit(x_train_noise, x_train, epochs=1, batch_size=32, verbose=1,
              validation_data=(x_test_noise, x_test))

    encoded = encoder.predict(x_test_noise[:10])
    decoded = decoder.predict(encoded)

    show_images(decoded)

    p = model.predict(x_test_noise[:10])
    show_images(p)
    plt.show()


denoise_auto_encoder()