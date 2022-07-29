# VAE_1.py
from tensorflow import keras
from keras.api._v2 import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def show_mean():
    m = keras.metrics.Mean()

    for i in range(10):
        m.update_state([i])
        print(m.result().numpy)
    print()

    print(m.count.numpy())      # 10.0
    print(m.total.numpy())      # 45.0
    print(m.built)              # True
    print(m.get_config())       # {'name': 'mean', 'dtype': 'float32'}


def get_mnist_concat():
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    print(x_train.shape)            # (60000, 28, 28)

    x = np.concatenate([x_train, x_test], axis=0)       # (70000, 28, 28)
    # x = np.expand_dims(x, axis=-1)
    # x = x.reshape(-1, 28, 28, 1)
    x = x[:, :, :, np.newaxis]
    # print(x.shape)

    return x / 255

def make_samples(mean, log_var):
    epsilon = keras.backend.random_normal(shape=tf.shape(mean))

    return mean + tf.exp(0.5 * log_var) * epsilon


def show_latent_space(model_path, x1, x2):
    model = keras.models.load_model(model_path)

    decoder = model.get_layer(index=1)
    fake = decoder.predict([[x1, x2]])
    # print(fake.shape)                                 # (1, 28, 28, 1)

    plt.title('({}, {})'.format(x1, x2))
    plt.imshow(fake.reshape(28, 28), cmap='gray')
    plt.tight_layout()
    plt.show()

def plot_label_clusters(model_path):
    model = keras.model.load_model(model_path)

    encoder = model.get_layer(index=0)
    (x_train, y_label), (_, _) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255

    mean, _, _ = encoder.predict(x_train)

    plt.figure(figsize=[8, 6])
    plt.scatter(mean[:, 0], mean[:, 1], c=y_label, s=3)
    plt.colorbar()
    plt.show()

# show_mean()
# get_mnist_concat()