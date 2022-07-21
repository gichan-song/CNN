# 19_1_lenet5_functional.py
from tensorflow import keras
from keras.api._v2 import keras

mnist = keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist
print(x_train.shape)                        # (60000, 28, 28)

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 퀴즈
# lenet5 시퀀셜 모델을 펑셔널 모델로 수정하세요
inputs = keras.layers.Input([28, 28, 1])
output = keras.layers.Conv2D(6, [5, 5], 1, 'same')(inputs)
output = keras.layers.MaxPool2D([2, 2], 2, 'same')(output)
output = keras.layers.Conv2D(16, [5, 5], 1, 'valid')(output)
output = keras.layers.MaxPool2D([2, 2], 2, 'valid')(output)
output = keras.layers.Flatten()(output)
output = keras.layers.Dense(120, activation='relu')(output)
output = keras.layers.Dense(84, activation='relu')(output)
output = keras.layers.Dense(10, activation='softmax')(output)

model = keras.Model(inputs, output)
model.summary()

model.compile(optimizer=keras.optimizers.SGD(0.01),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2,
          validation_data=(x_test, y_test))
