import unittest
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression


class LinearRegressionTest(unittest.TestCase):

    def test_simple(self):
        alpha = 0.00001
        max_iter = 100

        traing_x = np.arange(0, 300)
        traing_y = np.arange(1, 301)

        reg = LinearRegression(alpha, max_iter)

        loss = reg.fit(traing_x, traing_y)

        test_x = np.arange(20)
        test_y = np.arange(1, 21)
        pre_y = reg.predict(test_x)

        plt.subplot(211)
        plt.plot(np.arange(max_iter), loss)

        plt.subplot(212)
        plt.plot(test_x, test_y, 'o')
        plt.plot(test_x, pre_y)
        plt.annotate('$y=%.2f*x + %.2f$'%(reg.w, reg.b), (test_x[10], pre_y[10]), xytext=(test_x[10]-8, pre_y[10]+8), arrowprops={"arrowstyle":'->'})

        plt.show()

