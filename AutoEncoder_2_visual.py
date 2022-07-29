# AutoEncoder_2_visual.py
from tensorflow import keras
from keras.api._v2 import keras
import matplotlib.pyplot as plt
import numpy as np
import AutoEncoder_2


def show_prediction():
    x_train, y_train, x_test, y_test = AutoEncoder_2.get_mnist()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = keras.models.load_model('model/ae_mnist_conv_100.h5')

    # p = model.predict(x_train)
    # AutoEncoder_2.show_pca(p, y_train)

    # 텐서플로 버그라고 추측. 입력 크기와 배치 크기를 똑같이 만들면 정상 동작
    # p = model.predict(x_test, batch_size=10000)
    # AutoEncoder_2.show_pca(p, y_test)

    # 오토인코더 모델에는 레이어가 2개 있다
    # 0(인코더), 1(디코더)
    encoder = model.get_layer(index=0)
    encoder.summary()

    p = encoder.predict(x_train)
    # print(p.shape)                # (60000, 2)

    AutoEncoder_2.plot_label_clusters(p, y_train)


def show_images(samples):
    plt.figure(figsize=[20, 2])

    for i in range(10):
        ax = plt.subplot(1, 10, i+1)
        plt.imshow(samples[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()


def generate_number_image(size):
    x_train, y_train, _, _ = AutoEncoder_2.get_mnist()
    x_train = x_train.reshape(-1, 28, 28, 1)

    model = keras.models.load_model('model/ae_mnist_conv_100.h5')

    encoder = model.get_layer(index=0)
    decoder = model.get_layer(index=1)

    # 6만개 전체를 사용하면 좀더 나아질 수 있는 가능성 존재
    p = encoder.predict(x_train[:100])
    # AutoEncoder_2.plot_label_clusters(p, y_train[:100])
    # print(p.shape)                                    # (60000, 2)

    # -------------------------------------- #

    x_min, x_max = np.min(p[:, 0]), np.max(p[:, 0])
    y_min, y_max = np.min(p[:, 1]), np.max(p[:, 1])

    xx = np.random.uniform(x_min, x_max, size)
    yy = np.random.uniform(y_min, y_max, size)
    noises = np.transpose([xx, yy])
    # print(noises.shape)                               # (10, 2)
    # print(noises)

    # 퀴즈
    # noises가 만들어내는 이미지를 그래프에 그려보세요
    fake = decoder.predict(noises)
    # print(fake.shape)                                 # (10, 28, 28, 1)

    show_images(fake)


# 퀴즈
# 숫자 2개로 이루어진 좌표를 그림으로 변환하세요
def show_latent_space(x1, x2):
    model = keras.models.load_model('model/ae_mnist_conv_100.h5')

    decoder = model.get_layer(index=1)
    fake = decoder.predict([[x1, x2]])
    # print(fake.shape)                                 # (1, 28, 28, 1)

    plt.title('({}, {})'.format(x1, x2))
    plt.imshow(fake.reshape(28, 28), cmap='gray')
    plt.tight_layout()
    plt.show()


def show_latent_space_grid():
    model = keras.models.load_model('model/ae_mnist_conv_100.h5')
    decoder = model.get_layer(index=1)

    # 1번
    # grids = []
    # for r in range(-5, 6):
    #     for c in range(-5, 6):
    #         grids.append([r, c])
    #
    # grids = sorted(grids)
    # print(grids)
    #
    # x, y = [], []
    # for xx, yy in grids:
    #     x.append(xx)
    #     y.append(yy)
    #
    # print(x)
    # print(y)

    # 2번
    # x, y = np.meshgrid(range(-5, 6), range(-5, 6))
    # x, y = x.reshape(-1), y.reshape(-1)

    # 3번
    x = list(range(-5, 6)) * 11
    y = sorted(x)

    # plt.plot(x, y, 'ro')
    # plt.show()

    noises = np.transpose([x, y])
    fake = decoder.predict(noises)

    plt.figure(figsize=[20, 10])
    for i in range(11):
        for j in range(11):
            idx = i * 11 + j
            ax = plt.subplot(11, 11, idx+1)
            plt.imshow(fake[idx].reshape(28, 28), cmap='gray')

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.tight_layout()

    plt.tight_layout()
    plt.show()


# show_prediction()
# generate_number_image(size=10)

# show_latent_space(5, -15)
# show_latent_space(-15, 5)
# show_latent_space(0, 0)

show_latent_space_grid()

