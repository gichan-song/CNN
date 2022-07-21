# 6_1_linear_regression.py
import matplotlib.pyplot as plt

# ctrl + shift + f10
# alt + 1
# alt + 4


def cost(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        c += (hx - y[i]) ** 2

    return c / len(x)


def gradient_descent(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        # c += 2 * (hx - y[i]) ** (2 - 1) * (hx - y[i])미분
        # c += 2 * (hx - y[i]) * (w * x[i] - y[i])미분
        # c += 2 * (hx - y[i]) * x[i]
        c += (hx - y[i]) * x[i]

    return c / len(x)


def show_cost():
    # hx= wx + b
    # y = ax + b
    #     1    0
    # y =  x
    x = [1, 2, 3]
    y = [1, 2, 3]

    # print(cost(x, y, 0))
    # print(cost(x, y, 1))
    # print(cost(x, y, 2))

    for i in range(-30, 50):
        w = i / 10
        c = cost(x, y, w)
        print(w, c)

        plt.plot(w, c, 'ro')
    plt.show()


# 퀴즈
# 아래 코드를 w가 1.0이 되도록 수정하세요 (3가지)
def show_gradient():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = 5
    for i in range(100):
        c = cost(x, y, w)
        g = gradient_descent(x, y, w)
        w -= 0.1 * g

        print(i, w)


# show_cost()
show_gradient()


# 미분 : 기울기, 순간 변화량
#       x가 1만큼 변할 때 y가 변하는 정도

# y = 2x + 3

# y = 3                     3=1, 3=2, 3=3
# y = x                     1=1, 2=2, 3=3
# y = x + 1                 2=1, 3=2, 4=3
# y = (x + 1)
# y = 2x                    2=1, 4=2, 6=3
# y = xz

# y = x ^ 2                 1=1, 4=2, 9=3
#                           2 * x ^ (2 - 1) = 2x
#                           2 * x ^ (2 - 1) * x미분 = 2x
# y = (x + 1) ^ 2
#                           2 * (x + 1) ^ (2 - 1) = 2(x + 1)
#                           2 * (x + 1) ^ (2 - 1) * (x + 1)미분 = 2(x + 1)

