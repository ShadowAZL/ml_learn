import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression


class LogisticRegressionTest(unittest.TestCase):

    def test_simple(self):
        raw_data = pd.read_csv('ex2data2.txt', names=['Test 1', 'Test 2'])


        alpha = 0.00001
        max_iter = 100

        column_nums = raw_data.shape[1]
        traing_x = np.array(raw_data.iloc[:, :column_nums-1].values)
        traing_y = np.array(raw_data.iloc[:, -1].values)

        reg = LogisticRegression(alpha, max_iter)

        loss = reg.fit(traing_x, traing_y)

        pre_y = [1 if i>0.5 else 0 for i in reg.predict(traing_x)]

        plt.subplot(211)
        plt.plot(np.arange(max_iter), loss)

        ax = plt.subplot(212)
        plt.plot(traing_y, pre_y, 'o')

        correct = [a == b for (a, b) in zip(traing_y, pre_y)]
        accuracy = sum(correct) / len(correct)
        plt.text(0.2, 0.8, '%.2f%%'%(accuracy*100), transform=ax.transAxes)

        plt.show()

