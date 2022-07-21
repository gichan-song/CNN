# 23_1_cifar10.py
from tensorflow import keras
from keras.api._v2 import keras


def model_vgg():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[32, 32, 3]))
    model.add(keras.layers.Conv2D(16, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(16, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))
    model.add(keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.summary()

    return model


# 퀴즈
# 구글넷에 있는 인셉션 a와 인셉션 b, 리덕션 a, 리덕션 b를 응용해서 모델을 구축하세요
def model_google():
    inputs = keras.layers.Input([32, 32, 3])

    # stem
    x = keras.layers.Conv2D(6, [3, 3], 1, 'same', activation='relu')(inputs)

    # inception-a
    x1 = keras.layers.Conv2D(10, [1, 1], 1, 'same', activation='relu')(inputs)

    x2 = keras.layers.Conv2D(3, [1, 1], 1, 'same', activation='relu')(inputs)
    x2 = keras.layers.Conv2D(10, [3, 3], 1, 'same', activation='relu')(x2)

    x3 = keras.layers.Conv2D(3, [1, 1], 1, 'same', activation='relu')(inputs)
    x3 = keras.layers.Conv2D(10, [3, 3], 1, 'same', activation='relu')(x3)
    x3 = keras.layers.Conv2D(10, [3, 3], 1, 'same', activation='relu')(x3)

    x4 = keras.layers.AvgPool2D([3, 3], 1, 'same')(inputs)
    x4 = keras.layers.Conv2D(10, [1, 1], 1, 'same', activation='relu')(x4)

    x = keras.layers.concatenate([x1, x2, x3, x4], axis=3)

    # reduction-a
    # 내일 합니다~

    x = keras.layers.Flatten()(x)

    outputs = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.summary()
    exit()

    return model


# 퀴즈
# cifar10 데이터셋에 대해 동작하는 모델을 만드세요 (정확도 60% 이상)
# (1) x, y 생성  (2) 모델 구축  (3) 학습  (4) 결과 보기 + 예측
cifar10 = keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar10
print(x_train.shape, x_test.shape)          # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)          # (50000, 1) (10000, 1)

x_train = x_train / 255
x_test = x_test / 255

y_train = y_train.reshape(-1)               # (50000,)
y_test = y_test.reshape(-1)                 # (10000,)
print(y_train[:5])                          # [6 9 9 4 1]

# (2) 모델 구축
# model = model_vgg()
model = model_google()

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

# (3) 학습
model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2,
          validation_data=[x_test, y_test])

# (4) 결과 보기 + 예측


