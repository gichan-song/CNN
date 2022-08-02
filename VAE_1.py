# VAE_1.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras
import numpy as np
import matplotlib.pyplot as plt


def show_mean():
    m = keras.metrics.Mean()

    for i in range(10):
        m.update_state([i])
        print(m.result().numpy())
    print()

    print(m.count.numpy())      # 10.0
    print(m.total.numpy())      # 45.0
    print(m.built)              # True
    print(m.get_config())       # {'name': 'mean', 'dtype': 'float32'}

    # fit -> batch_size
    # 600번 step
    # 1번 스텝    310  310
    # 2번 스텝    290  300
    # 3번 스텝    270  290


def get_mnist_concat():
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    # print(x_train.shape)      # (60000, 28, 28)

    # print(np.concatenate([x_train[:10000], x_test], axis=0).shape)    # (20000, 28, 28)
    # print(np.concatenate([x_train[:10000], x_test], axis=1).shape)    # (10000, 56, 28)
    # print(np.concatenate([x_train[:10000], x_test], axis=2).shape)    # (10000, 28, 56)

    x = np.concatenate([x_train, x_test], axis=0)       # (70000, 28, 28)
    x = np.expand_dims(x, axis=-1)
    # x = x[:, :, :, np.newaxis]
    # x = x.reshape(-1, 28, 28, 1)
    # print(x.shape)                                    # (70000, 28, 28, 1)

    return x / 255


# 평균과 분산을 받아서 샘플 이미지 반환
def make_samples(mean, log_var):
    #                                        (batch_size, latent_dim)
    # epsilon = keras.backend.random_normal(shape=[70000, 2])
    epsilon = keras.backend.random_normal(shape=tf.shape(mean))

    return mean + tf.exp(0.5 * log_var) * epsilon


# 오토인코더에서 가져온 내용을 일부 수정
def show_latent_space(decoder, x1, x2):
    fake = decoder.predict([[x1, x2]])
    # print(fake.shape)                                 # (1, 28, 28, 1)

    plt.title('({}, {})'.format(x1, x2))
    plt.imshow(fake.reshape(28, 28), cmap='gray')
    plt.tight_layout()
    plt.show()


def plot_latent_space(decoder, n_samples):
    figure = np.zeros([28 * n_samples, 28 * n_samples])
    grid = np.linspace(-1, 1, n_samples)

    # ----------------------

    for r, rx in enumerate(grid):
        for c, cx in enumerate(reversed(grid)):
            p = decoder.predict([[rx, cx]])         # (1, 28, 28, 1)
            p = p.squeeze()                         # (28, 28)
            figure[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = p

    # ----------------------

    plt.figure(figsize=[7, 7])

    x = np.arange(n_samples) * 28 + 14
    y = np.round(grid, 2)

    plt.xticks(x, y)
    plt.yticks(x, y[::-1])
    plt.imshow(figure, cmap='Greys_r')

    plt.tight_layout()
    plt.show()


# 오토인코더에서 가져온 내용을 일부 수정
def plot_label_clusters(encoder):
    (x_train, y_label), (_, _) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255

    mean, _, _ = encoder.predict(x_train)

    plt.figure(figsize=[8, 6])
    plt.scatter(mean[:, 0], mean[:, 1], c=y_label, s=3)
    plt.colorbar()
    plt.show()


def make_encoder(latent_dim):
    inputs = keras.layers.Input(shape=[28, 28, 1])
    x = keras.layers.Conv2D(32, [3, 3], 2, 'same', activation='relu')(inputs)
    x = keras.layers.Conv2D(64, [3, 3], 2, 'same', activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(16, activation='relu')(x)

    mean = keras.layers.Dense(latent_dim, name='mean')(x)
    log_var = keras.layers.Dense(latent_dim, name='log_var')(x)
    z = make_samples(mean, log_var)

    return keras.Model(inputs, [mean, log_var, z], name='encoder')


def make_decoder(latent_dim):
    inputs = keras.layers.Input(shape=[latent_dim])
    x = keras.layers.Dense(7 * 7 * 64, activation='relu')(inputs)
    x = keras.layers.Reshape([7, 7, 64])(x)
    x = keras.layers.Conv2DTranspose(64, [3, 3], 2, 'same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(32, [3, 3], 2, 'same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(1, [3, 3], 1, 'same', activation='sigmoid')(x)

    return keras.Model(inputs, x, name='decoder')


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            r_loss_i = keras.losses.binary_crossentropy(data, reconstruction)
            r_loss = tf.reduce_mean(tf.reduce_sum(r_loss_i, axis=[1, 2]))

            kl_loss_i = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss_i, axis=1))

            total_loss = r_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            'loss': self.total_loss_tracker.result(),
            'r_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }


def save_model(enc_path, dec_path):
    digits = get_mnist_concat()
    # print(digits.shape)                   # (70000, 28, 28, 1)

    encoder = make_encoder(latent_dim=2)
    # encoder.summary()

    decoder = make_decoder(latent_dim=2)
    # decoder.summary()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(0.001))
    vae.fit(digits, epochs=30, batch_size=128, verbose=2)

    encoder.save(enc_path)
    decoder.save(dec_path)


def plot_model(enc_path, dec_path):
    # encoder = keras.models.load_model(enc_path)
    # plot_label_clusters(encoder)

    decoder = keras.models.load_model(dec_path)
    # show_latent_space(decoder, 0, 0)
    plot_latent_space(decoder, n_samples=10)


# show_mean()
# get_mnist_concat()

# save_model('model/vae_mnist_30_encoder.h5', 'model/vae_mnist_30_decoder.h5')
plot_model('model/vae_mnist_30_encoder.h5', 'model/vae_mnist_30_decoder.h5')
