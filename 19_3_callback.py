# 19_3_callback.py
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras             # 파이참 텐서플로 자동완성 안될 때 사용
import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np


iris = pd.read_csv('data/iris.csv')

x = iris.values[:, :-1]
x = np.float32(x)

enc = preprocessing.LabelEncoder()
y = enc.fit_transform(iris.variety)

data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data

model = keras.Sequential()
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer=keras.optimizers.SGD(0.01),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

# history = keras.callbacks.History()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
checkpoint = keras.callbacks.ModelCheckpoint(filepath='model/iris_{epoch:02d}_{val_loss:.2f}.h5',
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=True,
                                             initial_value_threshold=0.7)
plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1)

model.fit(x_train, y_train, epochs=1000, verbose=2,
          callbacks=[early_stopping, checkpoint, plateau],
          validation_data=(x_test, y_test))
