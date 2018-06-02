import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

def plot_data():
    data = pd.read_csv('microchips.txt', names=['test1', 'test2', 'result'])

    positive = data[data['result'] == 1]
    negative = data[data['result'] == 0]

    plt.plot(positive['test1'], positive['test2'], 'k+', label='y = 1')
    plt.plot(negative['test1'], negative['test2'], 'yo', label='y = 0')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()
    plt.show()


def map_feature(x1, x2, degree=6):
    x = np.ones((x1.shape[0], 1))
    for i in range(1, degree+1):
        for j in range(i+1):
            tmp = (x1**j) * (x2**(i-j))
            x = np.hstack((x, tmp.reshape(-1, 1)))
    return x


def fitting():
    data = pd.read_csv('microchips.txt', names=['test1', 'test2', 'result'])
    x1 = data['test1']
    x2 = data['test2']
    y = data['result']

    x = map_feature(x1.values, x2.values)

    alpha = 0.1
    max_iter = 1500
    model = LogisticRegression(alpha, max_iter, 0)
    loss, _ = model.fit(x, y, False)

    p = model.predict(x, False)
    p = [1 if i>0.5 else 0 for i in p]
    tp = sum([1.0 for vp, vy in zip(p, y) if vp == vy and vy == 1])
    tn = sum([1.0 for vp, vy in zip(p, y) if vp == vy and vy == 0])
    fp = sum([1.0 for vp, vy in zip(p, y) if vp == 1 and vy == 0 ])
    fn = sum([1.0 for vp, vy in zip(p, y) if vp == 0 and vy == 1])
    print(tp, tn, fp, fn)
    print('Accuracy %.3f' % ((tp + tn)/(tp + tn + fp + fn)))
    print('Precision %.3f' % (tp/(tp + fp)))
    print('Recall %.3f' % (tp/(tp + fn)))

    plt.figure(figsize=(6, 8))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(1, max_iter+1), loss)

    plt.subplot(2, 1, 2)
    positive = data[data['result'] == 1]
    negative = data[data['result'] == 0]
    plt.plot(positive['test1'], positive['test2'], 'k+')
    plt.plot(negative['test1'], negative['test2'], 'yo')


    x1, x2 = np.mgrid[-1:1.5:50j, -1:1.5:50j]
    p = np.zeros((50, 50))
    for i in range(50):
        for j in range(50):
            x = map_feature(np.array([x1[i, j]]), np.array([x2[i, j]])).squeeze()
            p[i, j] = x.dot(model.w)
    plt.contour(x1, x2, p, [0])


    plt.show()

fitting()

