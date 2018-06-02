import numpy as np


class LinearRegression(object):

    def __init__(self, alpha, max_iter):
        """
        :param alpha: learning rate 
        :param max_iter: max num of iteration
        """
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, x, y, intercept_adapt=True):
        if intercept_adapt:
            if x.ndim == 1:
                x = x[:, np.newaxis]
                x = np.hstack((np.ones((x.shape[0], 1)), x))
        self.x = x
        self.y = y
        self.w = np.random.normal(1, 0.1, (self.x.shape[1],))
        self.loss = []
        self.w_list = []
        for _ in range(self.max_iter):
            dw = self.gradient(self.w)
            self.w = self.w - self.alpha * dw
            self.loss.append(self.cost(self.x, self.y))
            self.w_list.append(self.w)
        return self.loss, self.w_list

    def predict(self, x, intercept_adapt=True):
        if intercept_adapt:
            if x.ndim == 1:
                x = x[:, np.newaxis]
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x.dot(self.w)

    def gradient(self, w):
        err = self.predict(self.x, False) - self.y
        dw = self.x.T.dot(err) / self.y.shape[0]
        return dw

    def cost(self, x, y):
        return 0.5 * np.mean((self.predict(x, False) - y)**2)
