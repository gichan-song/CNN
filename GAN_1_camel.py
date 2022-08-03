# GAN_1_camel.py
from tensorflow import keras
from keras.api._v2 import keras
import numpy as np
import matplotlib.pyplot as plt


def make_generator():
    gen_dense_size = (2, 2, 128)

    inputs = keras.layers.Input(shape=(100,))       # 100: latent dim

    x = keras.layers.Dense(np.prod(gen_dense_size))(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Reshape(gen_dense_size)(x)

    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Conv2D(128, [5, 5], 1, 'same')(x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Conv2D(64, [5, 5], 1, 'same')(x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Conv2D(64, [5, 5], 1, 'same')(x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(1, [5, 5], 1, 'same', activation='sigmoid')(x)

    model = keras.Model(inputs, x)
    model.summary()

    return model


def make_discriminator():
    inputs = keras.layers.Input(shape=[28, 28, 1])

    x = keras.layers.Conv2D(64, [5, 5], 2, 'same')(inputs)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Conv2D(64, [5, 5], 2, 'same')(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Conv2D(128, [5, 5], 2, 'same')(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Conv2D(128, [5, 5], 2, 'same')(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)

    model = keras.Model(inputs, x)
    model.summary()

    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    return model


def make_adversarial(generator, discriminator):
    inputs = keras.layers.Input(shape=(100,))       # 100: latent dim
    outputs = generator(inputs)
    outputs = discriminator(outputs)

    model = keras.Model(inputs, outputs)

    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=6e-8),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    return model


def train_gan(batch_size):
    discriminator = make_discriminator()
    generator = make_generator()
    adversarial = make_adversarial(generator, discriminator)


train_gan(batch_size=64)




