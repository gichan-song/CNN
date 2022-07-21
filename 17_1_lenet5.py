# 17_1_lenet5.py
import numpy as np
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
import matplotlib.pyplot as plt
import collections

mnist = keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist
print(x_train.shape)                        # (60000, 28, 28)

x_train = x_train / 255
x_test = x_test / 255

# x_train = x_train.reshape(-1, 784)        # Dense 레이어
# x_test = x_test.reshape(-1, 784)
x_train = x_train.reshape(-1, 28, 28, 1)    # Convolution 레이어
x_test = x_test.reshape(-1, 28, 28, 1)

model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=[28, 28, 1]))
# model.add(keras.layers.Conv2D(filters=9, kernel_size=[5, 3], strides=[1, 1], padding='valid'))
# model.add(keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
model.add(keras.layers.Conv2D(6, [5, 5], 1, 'same'))
model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))
model.add(keras.layers.Conv2D(16, [5, 5], 1, 'valid'))
model.add(keras.layers.MaxPool2D([2, 2], 2, 'valid'))
# model.add(keras.layers.Reshape([3380]))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(120, activation='relu'))
model.add(keras.layers.Dense(84, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
# model.summary()

model.compile(optimizer=keras.optimizers.SGD(0.01),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

# model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2,
#           validation_data=[x_test, y_test])
model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2)

# 퀴즈
# 예측한 결과로부터 틀린 데이터를 찾아주세요
# 정답과 오답을 함께 출력해 주세요
p = model.predict(x_test, verbose=0)
p_arg = np.argmax(p, axis=1)
print(p_arg.shape, y_test.shape)            # (10000,) (10000,)

not_equals = (p_arg != y_test)
print(not_equals[:20])

p_false = p_arg[not_equals]
y_false = y_test[not_equals]
x_false = x_test[not_equals]
print(p_false.shape, y_false.shape, np.sum(not_equals))

print('y : ', y_false[:20])
print('p : ', p_false[:20])

# plt.figure(figsize=[10, 1])
# for i, img in enumerate(x_false[:10]):
#     # print(img.shape)
#     plt.subplot(1, 10, i+1)
#     ax = plt.gca()
#     ax.axes.xaxis.set_visible(False)
#     ax.axes.yaxis.set_visible(False)
#     plt.imshow(img)
#     plt.title('{}({})'.format(p_false[i], y_false[i]))
#
# plt.tight_layout()
# plt.show()

# 퀴즈
# 예측한 결과로부터 가장 많이 틀린 숫자를 알려주세요
freq = {}
for yy in y_false:
    if yy not in freq:
        freq[yy] = 0

    freq[yy] += 1

print(freq)
print(freq.items())
print(sorted(freq.items(), key=lambda t: t[1], reverse=True))

freq = collections.Counter(y_false)
print(freq)
print(freq.most_common())

# 이왕이면 where 함수를 사용하는 것이 좋습니다
# import numpy as np
# a = np.int32([3, 4, 5, 6, 7])
# b = np.int32([1, 3, 5, 7, 9])
#
# equals = (a == b)
# print(equals)
# print(a[equals])
# print(b[equals])
#
# pos = np.where(a != b)      # (array([2], dtype=int64),)
# pos = pos[0]
# print(pos)                  # [2]
#
# print(a[pos])
# print(b[pos])



