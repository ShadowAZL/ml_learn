import unittest
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression


class LinearRegressionTest(unittest.TestCase):

    def test_simple(self):
        alpha = 0.00001
        max_iter = 100

        traing_x = np.arange(0, 300).reshape((150, 2))
        traing_y = np.arange(1, 151)

        reg = LinearRegression(alpha, max_iter)

        loss = reg.fit(traing_x, traing_y)

        test_x = np.arange(20).reshape((10, 2))
        test_y = np.arange(1, 11)
        pre_y = reg.predict(test_x)

        plt.subplot(211)
        plt.plot(np.arange(max_iter), loss)

        plt.subplot(212)
        plt.plot(test_y, pre_y, 'o')

        plt.show()

