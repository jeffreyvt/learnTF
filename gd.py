
import numpy as np
import matplotlib.pyplot as plt


def F(x):
    return x**2


def grad_F(x):
    return 2*x



def gradient_descent(F, grad_F, learning_rate, x, i):
    count = 0
    x_val = [x]
    y_val = [F(x)]
    while count < i:
        x = x-learning_rate*grad_F(x)
        count += 1
        # print(F(x))
        x_val.append(x)
        y_val.append(F(x))
    return x_val[::-1], y_val[::-1]
x = np.linspace(-3,3,50)
y = x*x



x_val, y_val = gradient_descent(F, grad_F, 0.1, 2, 100)

plt.plot(x,y,"-",x_val, y_val, "-o")
plt.show()