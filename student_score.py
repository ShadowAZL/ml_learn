import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression, sigmoid


def plot_data():
    data = pd.read_csv('student_score.txt', names=['Exam1', 'Exam2', 'admission'])

    negative = data[data['admission'] == 0]
    positive = data[data['admission'] == 1]

    plt.plot(negative['Exam1'], negative['Exam2'], 'yo')
    plt.plot(positive['Exam1'], positive['Exam2'], 'k+')

    plt.show()


def cost():
    data = pd.read_csv('student_score.txt', names=['Exam1', 'Exam2', 'admission'])
    x = data[['Exam1', 'Exam2']]
    y = data['admission']
    w = np.zeros(3)
    x = np.hstack((np.ones((y.size, 1)), x))
    p = sigmoid(x.dot(w))
    cost = -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
    print('Cost %.3f when w is zero' % cost)


def fitting():
    data = pd.read_csv('student_score.txt', names=['Exam1', 'Exam2', 'admission'])
    x = data[['Exam1', 'Exam2']]
    y = data['admission']

    print(x.mean())
    print(x.max() - x.min())

    x = (x - x.mean())/(x.max() - x.min())

    alpha = 10
    max_iter = 150
    model = LogisticRegression(alpha, max_iter)
    loss, _ = model.fit(x, y)

    p = model.predict(np.array([[1, (45.0 - 65.644274)/69.769035, (85.0 - 66.221998)/68.266173]]), False)
    print('Predict %.3f when Exam1 euqals 45 and Exam2 equals 85' % p)


    plt.subplot(2, 1, 1)
    plt.plot(np.arange(1, max_iter+1), loss)
    plt.title('Loss Curve')

    plt.subplot(2, 1, 2)
    negative = data[data['admission'] == 0]
    positive = data[data['admission'] == 1]
    plt.plot(negative['Exam1'], negative['Exam2'], 'yo')
    plt.plot(positive['Exam1'], positive['Exam2'], 'k+')

    print(model.w)

    bx = data['Exam1']
    by = (-68.266173/model.w[2]) * (((bx - 65.644274)/69.769035)*model.w[1] + model.w[0]) + 66.221998

    x = data[['Exam1', 'Exam2']]
    x = (x - x.mean()) / (x.max() - x.min())

    p = [1 if i >= 0.5 else 0 for i in model.predict(x)]
    tp = sum([1.0 for vp, vy in zip(p, y) if vp == vy and vy == 1])
    tn = sum([1.0 for vp, vy in zip(p, y) if vp == vy and vy == 0])

    fp = sum([1.0 for vp, vy in zip(p, y) if vp == 1 and vy == 0])
    fn = sum([1.0 for vp, vy in zip(p, y) if vp == 0 and vy == 1])

    print(tp, tn, fp, fn)
    print('Accurancy %.2f' % ((tp + tn)/(tp + tn + fp + fn)))
    print('Precision %.2f' % (tp/(tp + fp)))
    print('Recall %.2f' % (tp/(tp + fn)))

    plt.plot(bx, by)

    plt.show()

if __name__ == '__main__':
    fitting()

