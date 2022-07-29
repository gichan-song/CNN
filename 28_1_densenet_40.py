# 28_1_densenet_40.py
from tensorflow import keras
from keras.api._v2 import keras
import numpy as np


def get_cifar10():
    cifar10 = keras.datasets.cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = cifar10

    # print(x_train.dtype)                      # uint8
    # print(x_train.shape)                      # (50000, 32, 32, 3)
    # print(np.min(x_train), np.max(x_train))   # 0 255

    x_train = keras.applications.densenet.preprocess_input(x_train)
    x_test  = keras.applications.densenet.preprocess_input(x_test )
    # print(x_train.dtype)                      # float32
    # print(x_train.shape)                      # (50000, 32, 32, 3)
    # print(np.min(x_train), np.max(x_train))   # -2.117904 2.64

    return x_train, y_train, x_test, y_test


def bn_relu_conv(x, n_filters, kernel_size, has_bottleneck):
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    if has_bottleneck:
        x = keras.layers.Conv2D(n_filters * 4, 1, 1, 'same',
                                kernel_initializer='he_normal',
                                use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

    return keras.layers.Conv2D(n_filters, kernel_size, 1, 'same',
                               kernel_initializer='he_normal',
                               use_bias=False)(x)


def dense_block(x, n_layers, growth_rate, has_bottleneck):
    for _ in range(n_layers):
        c = bn_relu_conv(x, growth_rate, [3, 3], has_bottleneck)
        x = keras.layers.concatenate([x, c])

    return x


def transition_layer(x, n_filters):
    x = bn_relu_conv(x, n_filters, [1, 1], has_bottleneck=False)
    return keras.layers.AvgPool2D([2, 2], 2)(x)


def densenet_40(input_shape, has_bottleneck, compression):
    inputs = keras.layers.Input(input_shape)
    x = inputs

    # ------------------------------------------------ #

    depth, growth_rate = 40, 12
    n_layers = (depth - 4) // 3

    # ------------------------------------------------ #

    n_filters = growth_rate * 2
    x = keras.layers.Conv2D(n_filters, [3, 3], 1, 'same',
                            kernel_initializer='he_normal',
                            use_bias=False)(x)

    # ------------------------------------------------ #

    x = dense_block(x, n_layers, growth_rate, has_bottleneck)

    n_filters += growth_rate * n_layers
    n_filters = int(n_filters * compression)

    x = transition_layer(x, n_filters)

    # ------------------------------------------------ #

    x = dense_block(x, n_layers, growth_rate, has_bottleneck)

    n_filters += growth_rate * n_layers
    n_filters = int(n_filters * compression)

    x = transition_layer(x, n_filters)

    # ------------------------------------------------ #

    x = dense_block(x, n_layers, growth_rate, has_bottleneck)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.GlobalAvgPool2D()(x)
    x = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, x)
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    return model


model = densenet_40([32, 32, 3], has_bottleneck=False, compression=1.0)
model_path = 'model/densenet_40.h5'

# model = densenet_40([32, 32, 3], has_bottleneck=True, compression=1.0)
# model_path = 'model/densenet_40_b.h5'
#
# model = densenet_40([32, 32, 3], has_bottleneck=False, compression=0.5)
# model_path = 'model/densenet_40_c.h5'
#
# model = densenet_40([32, 32, 3], has_bottleneck=True, compression=0.5)
# model_path = 'model/densenet_40_bc.h5'

x_train, y_train, x_test, y_test = get_cifar10()

gen_train = keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                         rotation_range=15,
                                                         width_shift_range=5/32,
                                                         height_shift_range=5/32,
                                                         horizontal_flip=True)
gen_test = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

flow_train = gen_train.flow(x_train, y_train, batch_size=64)
flow_test = gen_test.flow(x_test, y_test, batch_size=64)

model.fit(flow_train, epochs=100, validation_data=flow_test)







