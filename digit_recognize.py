import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

import random
from logistic_regression import LogisticRegression


def plot_data():
    data = loadmat('digit_recognize.mat')
    x = data['X']

    indexs = np.random.randint(0, 5000, size=100)

    height, width = 20, 20
    img = np.zeros((10*height, 10*width))
    c_row, c_col = 0, 0
    for i in indexs:
        dig = x[i].reshape(height, width)
        ix, iy = c_row*height, c_col*width
        img[ix:ix+height, iy:iy+height] = dig

        c_col += 1
        if c_col == 10:
            c_row += 1
            c_col = 0

    plt.imshow(img.T, cmap=plt.get_cmap('Greys_r'))
    plt.xticks(())
    plt.yticks(())

    plt.show()


def fitting():
    data = loadmat('digit_recognize.mat')
    x = data['X']
    y = data['y']

    alpha = 5
    max_iter = 100
    lamb = 0.5
    models = []

    _, axes = plt.subplots(5, 2, figsize=(6, 7))
    axes = np.ravel(axes)

    for i in range(10):

        model = LogisticRegression(alpha, max_iter, lamb)
        if i == 0:
            yi = np.array([1 if d == 10 else 0 for d in y])
        else:
            yi = np.array([1 if d == i else 0 for d in y])
        loss, _ = model.fit(x, yi)
        models.append(model)

        axes[i].plot(np.arange(1, max_iter+1), loss)
        axes[i].set_title('%d' % i)

    p = []
    for i in range(10):
        p.append(models[i].predict(x).reshape(-1, 1))
    p = np.hstack(p)
    p = np.argmax(p, axis=1)

    acu = sum([1.0 if vp == vy or (vp == 0 and vy == 10) else 0 for vp, vy in zip(p, y)])
    print('Accuracy %.4f' % (acu/5000.0))

    plt.subplots_adjust(hspace=0.9)
    plt.show()

fitting()
