# 27_2_feature_map.py
import numpy as np
from tensorflow import keras
from keras.api._v2 import keras
import matplotlib.pyplot as plt


def load_image(img_path, target_size):
    img = keras.preprocessing.image.load_img(img_path,
                                             target_size=[target_size, target_size])
    # print(type(img))            # <class 'PIL.Image.Image'>
    img_tensor = keras.preprocessing.image.img_to_array(img)
    # print(type(img_tensor))     # <class 'numpy.ndarray'>
    # print(img_tensor.shape)     # (64, 64, 3)

    return img_tensor[np.newaxis] / 255     # (1, 64, 64, 3)


def show_image(img_path, target_size):
    img = load_image(img_path, target_size)
    img = img.reshape(img.shape[1:])        # (64, 64, 3)

    plt.imshow(img)
    plt.show()


def show_first_activation_map(model_path, img_path):
    old_model = keras.models.load_model(model_path)

    outputs = old_model.layers[0].output
    print(outputs.shape)                            # (None, 64, 64, 16)

    model = keras.Model(old_model.input, outputs)

    img = load_image(img_path, target_size=64)
    print(img.shape)                                # (1, 64, 64, 3)

    feature_maps = model.predict(img, verbose=0)
    print(feature_maps.shape)                       # (1, 64, 64, 16)

    plt.imshow(feature_maps[0, :, :, 0], cmap='gray')
    plt.show()


def predict_and_get_outputs(model_path, img_path):
    old_model = keras.models.load_model(model_path)

    outputs = [layer.output for layer in old_model.layers[:9]]
    names = [layer.name for layer in old_model.layers[:9]]
    # print([str(output.shape) for output in outputs])
    # ['(None, 64, 64, 16)', '(None, 64, 64, 16)', '(None, 32, 32, 16)', '(None, 32, 32, 32)', '(None, 32, 32, 32)',
    #  '(None, 16, 16, 32)', '(None, 16, 16, 32)', '(None, 16, 16, 32)', '(None, 8, 8, 32)']

    model = keras.Model(old_model.input, outputs)

    img = load_image(img_path, target_size=64)
    result = model.predict(img, verbose=0)

    return result, names


def show_activation_map(layer, title):
    size, n_features = layer.shape[1], layer.shape[-1]

    cols = 8
    rows = n_features // cols

    big_image = np.zeros([rows * size, cols * size], dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            channel = layer[0, :, :, r * cols + c]

            channel -= channel.mean()
            channel /= channel.std()
            channel *= 64
            channel += 128
            channel = np.clip(channel, 0, 255).astype('uint8')

            big_image[r * size:(r+1) * size, c * size:(c+1) * size] = channel

    plt.figure(figsize=[int(cols * 2 * (size / 64)), int(rows * 2 * (size / 64))])
    plt.xticks(np.arange(cols) * size)
    plt.yticks(np.arange(rows) * size)
    plt.title(title)
    plt.imshow(big_image)                       # cmap='gray'
    plt.tight_layout()


model_path = 'model/model_cats_and_dogs_2.h5'
img_path = 'cats_and_dogs/train/cat.500.jpg'
img_path = 'cats_and_dogs/train/dog.500.jpg'

show_image(img_path, 64)
# show_first_activation_map(model_path, img_path)

# 퀴즈
# 이미지 1장에 대해 예측한 9개의 결과를 모두 그래프로 그려주세요
outputs, titles = predict_and_get_outputs(model_path, img_path)
# show_activation_map(outputs[0], titles[0])

for i in range(len(outputs)):
    show_activation_map(outputs[i], titles[i])

plt.show()

















