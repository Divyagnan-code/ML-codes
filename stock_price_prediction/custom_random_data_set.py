import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use("ggplot")


def create_random_dataset(n_patterns, variance, step=2, corr=False):
    val = 1
    ys = []
    for i in range(n_patterns):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if corr and corr == "pos":
            val += step
        elif corr and corr == "neg":
            val -= step
    xs = [i for i in range(n_patterns)]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit(xs, ys):
    m = ((mean(xs) * mean(ys)) - (mean(xs * ys)))/((mean(xs) * mean(xs)) - (mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b


def squared_error(ys_orig, ys_line):
    return sum((ys_orig - ys_line) ** 2)


def coeff_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_reg = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_reg / squared_error_y_mean)


xs, ys = create_random_dataset(200, 40, step=1, corr='pos')
m, b = best_fit(xs, ys)
line_reg = [(m*x)+b for x in xs]
predict_x = 300
predict_y = m*300+b

r_squared = coeff_of_determination(ys, line_reg)
print(r_squared)
plt.scatter(xs, ys)
plt.plot(xs, line_reg)
plt.show()
