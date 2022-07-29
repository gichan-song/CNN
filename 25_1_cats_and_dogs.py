# 25_1_cats_and_dogs.py
from tensorflow import keras
from keras.api._v2 import keras
import pickle
import matplotlib.pyplot as plt


def save_history(history, history_path):
    f = open(history_path, 'wb')
    pickle.dump(history.history, f)
    f.close()


def load_history(history_path):
    f = open(history_path, 'rb')
    history = pickle.load(f)
    f.close()
    return history


def plot_history(history):
    plt.figure()
    x = range(len(history['acc']))

    plt.subplot(1, 2, 1)
    plt.plot(x, history['acc'], 'r', label='train')
    plt.plot(x, history['val_acc'], 'g', label='valid')
    plt.ylim(0, 1)
    plt.title('acc')
    plt.legend()

    # 퀴즈
    # loss 그래프를 그려주세요
    plt.subplot(1, 2, 2)
    plt.plot(x, history['loss'], 'r', label='train')
    plt.plot(x, history['val_loss'], 'g', label='valid')
    plt.ylim(0, 1)
    plt.title('loss')
    plt.legend()


# 퀴즈
# train, validation, test 폴더에 대해 제너레이터를 만들고
# 폴더로부터 이미지를 읽어올 수 있는 준비 코드를 만드세요 (이미지 크기 32)
def model_1(img_size, batch_size):
    gen_train = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    flow_train = gen_train.flow_from_directory('cats_and_dogs/root/train',
                                               target_size=[img_size, img_size],
                                               batch_size=batch_size,
                                               class_mode='binary')

    gen_valid = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    flow_valid = gen_valid.flow_from_directory('cats_and_dogs/root/validation',
                                               target_size=[img_size, img_size],
                                               batch_size=batch_size,
                                               class_mode='binary')

    gen_test = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    flow_test = gen_test.flow_from_directory('cats_and_dogs/root/test',
                                             target_size=[img_size, img_size],
                                             batch_size=batch_size,
                                             class_mode='binary')

    # 퀴즈
    # 제너레이터를 사용하는 모델을 만드세요
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[img_size, img_size, 3]))
    model.add(keras.layers.Conv2D(16, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(16, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))
    model.add(keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))
    model.add(keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=keras.optimizers.RMSprop(0.0001),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    history = model.fit(flow_train, epochs=10, verbose=2, validation_data=flow_valid)
    model.evaluate(flow_test, verbose=0)

    save_history(history, 'model/model_cats_and_dogs_1.history')
    model.save('model/model_cats_and_dogs_1.h5')


# 퀴즈
# 1번 코드에서 제너레이터에 들어가는 이미지 증강 옵션을 사용해서 모델을 검증하세요
def model_2(img_size, batch_size):
    gen_train = keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                             horizontal_flip=True,
                                                             rotation_range=50,
                                                             zoom_range=(0.5, 3.0))
    flow_train = gen_train.flow_from_directory('cats_and_dogs/root/train',
                                               target_size=[img_size, img_size],
                                               batch_size=batch_size,
                                               class_mode='binary')

    gen_valid = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    flow_valid = gen_valid.flow_from_directory('cats_and_dogs/root/validation',
                                               target_size=[img_size, img_size],
                                               batch_size=batch_size,
                                               class_mode='binary')

    flow_test = gen_valid.flow_from_directory('cats_and_dogs/root/test',
                                              target_size=[img_size, img_size],
                                              batch_size=batch_size,
                                              class_mode='binary')

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[img_size, img_size, 3]))
    model.add(keras.layers.Conv2D(16, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(16, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))
    model.add(keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))
    model.add(keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(32, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], 2, 'same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=keras.optimizers.RMSprop(0.0001),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    history = model.fit(flow_train, epochs=10, verbose=2, validation_data=flow_valid)
    model.evaluate(flow_test, verbose=0)

    save_history(history, 'model/model_cats_and_dogs_2.history')
    model.save('model/model_cats_and_dogs_2.h5')


# model_1(img_size=64, batch_size=32)
# model_2(img_size=64, batch_size=32)

h1 = load_history('model/model_cats_and_dogs_1.history')
h2 = load_history('model/model_cats_and_dogs_2.history')

plot_history(h1)
plot_history(h2)
plt.show()

