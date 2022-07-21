# 20_3_adult.py
from tensorflow import keras
from keras.api._v2 import keras
import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np


def make_xy_1():
    enc = preprocessing.LabelEncoder()
    workclass = enc.fit_transform(adult['workclass'])
    education = enc.fit_transform(adult['education'])
    marital_status = enc.fit_transform(adult['marital-status'])
    occupation = enc.fit_transform(adult['occupation'])
    relationship = enc.fit_transform(adult['relationship'])
    race = enc.fit_transform(adult['race'])
    sex = enc.fit_transform(adult['sex'])
    native_country = enc.fit_transform(adult['native-country'])
    # print(workclass[:10])                     # [7 6 4 4 4 4 4 6 4 4]

    x = [adult['age'].values, adult['fnlwgt'].values, adult['education-num'].values,
         adult['capital-gain'].values, adult['capital-loss'].values, adult['hours-per-week'].values,
         workclass, education, marital_status, occupation, relationship, race, sex, native_country]

    print(np.int32(x).shape)  # (14, 32561)
    x = np.transpose(x)

    income = enc.fit_transform(adult['income'])  # (32561,)
    y = income.reshape(-1, 1)
    # print(x.shape, y.shape)  # (32561, 14) (32561, 1)

    return x, y


# 퀴즈
# make_xy_1 함수에 사용한 LabelEncoder 클래스를 LabelBinarizer 클래스로 교체하세요
def make_xy_2():
    enc = preprocessing.LabelBinarizer()
    workclass = enc.fit_transform(adult['workclass'])
    education = enc.fit_transform(adult['education'])
    marital_status = enc.fit_transform(adult['marital-status'])
    occupation = enc.fit_transform(adult['occupation'])
    relationship = enc.fit_transform(adult['relationship'])
    race = enc.fit_transform(adult['race'])
    sex = enc.fit_transform(adult['sex'])
    native_country = enc.fit_transform(adult['native-country'])
    # print(workclass.shape)                      # (32561, 9)

    x = [adult['age'].values, adult['fnlwgt'].values, adult['education-num'].values,
         adult['capital-gain'].values, adult['capital-loss'].values, adult['hours-per-week'].values]
    x = np.transpose(x)
    # print(x.shape)                              # (32561, 6)

    # h: horizontal, v: vertical
    x = np.hstack([x, workclass, education, marital_status,
                   occupation, relationship, race, sex, native_country])

    y = enc.fit_transform(adult['income'])
    # print(x.shape, y.shape)                     # (32561, 107) (32561, 1)

    return x, y


# 퀴즈
# adult.data 파일을 읽어서
# 80%로 학습하고 20%에 대해 결과를 구하세요
names = 'age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income'
adult = pd.read_csv('data/adult.data',
                    names=names.split(','))
# print(adult)                                # [32561 rows x 15 columns]
# print(adult.describe())
# adult.info()

# 목수
# LabelEncoder: 7
# LabelBinarizer: 0 0 0 0 0 0 0 1 0
# x, y = make_xy_1()
x, y = make_xy_2()

# x = preprocessing.scale(x)          # 표준화
x = preprocessing.minmax_scale(x)

# 데이터 분할 -> 모델 구축 -> 스케일링 -> 하이퍼 파라미터 조절 -> 86%
data = model_selection.train_test_split(x, y, test_size=10000)
x_train, x_test, y_train, y_test = data
# print(x_train.shape, x_test.shape)          # (22561, 14) (10000, 14)
# print(y_train.shape, y_test.shape)          # (22561, 1) (10000, 1)

model = keras.Sequential()
model.add(keras.layers.InputLayer(x.shape[1:]))         # (14,)
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(12, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.binary_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2,
          validation_data=(x_test, y_test))
