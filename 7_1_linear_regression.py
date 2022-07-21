# 7_1_linear_regression.py
import tensorflow as tf

def linear_regression_1():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(5.0)
    b = tf.Variable(-3.0)

    optimizer = tf.keras.optimizers.SGD(0.1)            # Stochastic Gradient Descent

    for i in range(10):
        with tf.GradientTape() as tape:
            hx = w * x + b
            loss = tf.reduce_mean((hx - y) ** 2)

        gradient = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(gradient, [w, b]))

        print(i, loss.numpy())
    print()

    # 퀴즈
    # x가 5와 7의 값을 가질 때 y를 예측하세요
    print('5 :', (w * 5 + b).numpy())
    print('7 :', (w * 7 + b).numpy())
    print()

    print('5 :', w.numpy() * 5 + b.numpy())
    print('7 :', w.numpy() * 7 + b.numpy())
    print()

    print('* :', w * x + b)
    print('* :', w * [1, 2, 3] + b)
    print('* :', w * [5, 7] + b)



    # a = [1, 2, 3]
    # b = ['x', 'y', 'z']
    #
    # for i in (a, b):
    #     print(i)
    #
    # for i in zip(a, b):
    #     print(i)


def linear_regression_2():
    def predict(x,w,b):
        return w*x+b

    def mean_square_error(y,p):
        return tf.reduce_mean((p - y) ** 2)
    x = [1, 2, 3] #feature
    y = [1, 2, 3] #target

    w = tf.Variable(tf.random.uniform([1]))
    b = tf.Variable(tf.random.uniform([1]))

    optimizer = tf.keras.optimizers.SGD(0.1)  # Stochastic Gradient Descent

    for i in range(10):
        with tf.GradientTape() as tape:
            hx = predict(x,w,b)
            loss = mean_square_error(y,hx)

        gradient = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(gradient, [w, b]))

        print(i, loss.numpy())
    print(predict([5,7],w,b).numpy())

linear_regression_2()