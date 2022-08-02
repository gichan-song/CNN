# VAE_2.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing


def make_xy_sparse():
    leaf = pd.read_csv('data/leaf_train.csv', index_col=0)
    # print(leaf)

    x = leaf.values[:, 1:].astype(np.float32)
    species = leaf.values[:, 0]

    x = preprocessing.scale(x)

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(species)

    return x, y


class CustomModel_1(keras.Model):
    def train_step(self, data):
        # print(type(data), len(data))    # <class 'tuple'> 2
        # xx, yy = data
        # print(xx.shape, yy.shape)       # (None, 192) (None,)

        return super().train_step(data)

    def test_step(self, data):
        # print(type(data), len(data))    # <class 'tuple'> 2
        # xx, yy = data
        # print(xx.shape, yy.shape)       # (None, 192) (None,)
        return super().test_step(data)

    def predict_step(self, data):
        return super().predict_step(data)


class CustomModel_2(keras.Model):
    def train_step(self, data):
        xx, yy = data
        with tf.GradientTape() as tape:
            y_pred = super().__call__(xx, training=True)
            loss = self.compiled_loss(yy, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(yy, y_pred)
        return {m.name: m.result() for m in self.metrics}


class CustomModel_3(keras.Model):
    def __init__(self, model, **kwargs):
        super(CustomModel_3, self).__init__(kwargs)
        self.model = model
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.acc_metric = keras.metrics.SparseCategoricalCrossentropy(name='acc')

    def train_step(self, data):
        xx, yy = data
        with tf.GradientTape() as tape:
            y_pred = self.model(xx, training=True)
            loss = keras.losses.sparse_categorical_crossentropy(yy, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.acc_metric.update_state(yy, y_pred)

        return {'loss': self.loss_tracker.result(), 'acc': self.acc_metric.result()}


def softmax_regression_leaf_1():
    x, y = make_xy_sparse()
    print(x.shape)              # (990, 192)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8)

    inputs = keras.layers.Input(shape=[192])
    outputs = keras.layers.Dense(99, activation='softmax')(inputs)

    # model = keras.Model(inputs, outputs)
    # model = CustomModel_1(inputs, outputs)
    model = CustomModel_2(inputs, outputs)

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')
    model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=2,
              validation_data=(x_test, y_test))


def softmax_regression_leaf_2():
    x, y = make_xy_sparse()
    print(x.shape)              # (990, 192)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8)

    model_base = keras.Sequential([
        keras.layers.InputLayer(input_shape=[192]),
        keras.layers.Dense(99, activation='softmax'),
    ])

    model = CustomModel_3(model_base)
    model.compile(optimizer=keras.optimizers.Adam(0.001))

    # validation_data 옵션을 사용하면 커스텀 모델에 test_step 함수도 오버라이딩해야 합니다
    model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=2)


# softmax_regression_leaf_1()
softmax_regression_leaf_2()
