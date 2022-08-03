# VAE_5_CelebA.py
# VAE_4_CelebA.py 파일을 복사해서 일부 수정
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras
import numpy as np
import matplotlib.pyplot as plt


def get_npy_images(npy_path):
    images = np.load(npy_path)
    return images / 255


# 평균과 분산을 받아서 샘플 이미지 반환
def make_samples(mean, log_var):
    #                                        (batch_size, latent_dim)
    # epsilon = keras.backend.random_normal(shape=[70000, 2])
    epsilon = keras.backend.random_normal(shape=tf.shape(mean))

    return mean + tf.exp(0.5 * log_var) * epsilon


# 오토인코더에서 가져온 내용을 일부 수정
def show_latent_space(decoder, x1, x2):
    fake = decoder.predict([[x1, x2]])
    # print(fake.shape)                                 # (1, 64, 64, 3)

    plt.title('({}, {})'.format(x1, x2))
    plt.imshow(fake.squeeze())                          # (64, 64, 3)
    plt.tight_layout()
    plt.show()


def plot_latent_space(decoder, n_samples):
    figure = np.zeros([64 * n_samples, 64 * n_samples, 3])      # 3: 컬러
    grid = np.linspace(-10, 10, n_samples)                      # (-1, 1)을 (-10, 10)으로 확장

    # ----------------------

    for r, rx in enumerate(grid):
        for c, cx in enumerate(reversed(grid)):
            p = decoder.predict([[rx, cx]])         # (1, 64, 64, 3)
            p = p.squeeze()                         # (64, 64, 3)
            figure[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = p

    # ----------------------

    plt.figure(figsize=[7, 7])

    x = np.arange(n_samples) * 64 + 32
    y = np.round(grid, 2)

    plt.xticks(x, y)
    plt.yticks(x, y[::-1])
    plt.imshow(figure)

    plt.tight_layout()
    plt.show()


# 오토인코더에서 가져왔지만, 셀럽A에서는 정답이 없기 때문에 사용할 수 없다
# 이 함수의 목적은 정답 분포를 보여주는 것이기 때문에.
def plot_label_clusters(encoder):
    pass


def make_encoder(latent_dim):
    inputs = keras.layers.Input(shape=[64, 64, 3])
    x = keras.layers.Conv2D(32, [3, 3], 2, 'same', activation='relu')(inputs)
    x = keras.layers.Conv2D(64, [3, 3], 2, 'same', activation='relu')(x)
    x = keras.layers.Conv2D(128, [3, 3], 2, 'same', activation='relu')(x)       # 추가한 레이어
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(16, activation='relu')(x)

    mean = keras.layers.Dense(latent_dim, name='mean')(x)
    log_var = keras.layers.Dense(latent_dim, name='log_var')(x)
    z = make_samples(mean, log_var)

    return keras.Model(inputs, [mean, log_var, z], name='encoder')


def make_decoder(latent_dim):
    inputs = keras.layers.Input(shape=[latent_dim])
    x = keras.layers.Dense(8 * 8 * 64, activation='relu')(inputs)
    x = keras.layers.Reshape([8, 8, 64])(x)
    x = keras.layers.Conv2DTranspose(128, [3, 3], 2, 'same', activation='relu')(x)       # 추가한 레이어
    x = keras.layers.Conv2DTranspose(64, [3, 3], 2, 'same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(32, [3, 3], 2, 'same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(3, [3, 3], 1, 'same', activation='sigmoid')(x)

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

    # loss tracker가 에포크 시작할 때 자동으로 초기화되도록 만들어 줌
    # self.loss_tracker.reset_state()
    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]


def save_model(enc_path, dec_path):
    faces = get_npy_images('celeb_a/train_1000.npy')
    # print(faces.shape)                   # (1000, 64, 64, 3)

    encoder = make_encoder(latent_dim=2)
    # encoder.summary()

    decoder = make_decoder(latent_dim=2)
    # decoder.summary()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(0.001))
    vae.fit(faces, epochs=30, batch_size=128, verbose=2)

    encoder.save(enc_path)
    decoder.save(dec_path)


def plot_model(enc_path, dec_path):
    # encoder = keras.models.load_model(enc_path)
    # plot_label_clusters(encoder)

    decoder = keras.models.load_model(dec_path)
    # show_latent_space(decoder, 0, 0)
    plot_latent_space(decoder, n_samples=10)


# 퀴즈
# make_encoder 함수에 추가한 레이어에 대해
# make_decoder 함수에도 쌍을 이루는 레이어를 추가하고 결과를 확인하세요
save_model('model/vae_celeba_30_encoder.h5', 'model/vae_celeba_30_decoder.h5')
plot_model('model/vae_celeba_30_encoder.h5', 'model/vae_celeba_30_decoder.h5')
