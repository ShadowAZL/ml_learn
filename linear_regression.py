import numpy as np


class LinearRegression(object):

    def __init__(self, alpha, max_iter):
        """
        :param alpha: learning rate 
        :param max_iter: max num of iteration
        """
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        self.x = np.hstack((np.ones((x.shape[0], 1)), x))
        self.y = y

        self.w = np.random.normal(1, 0.1, (self.x.shape[1],))
        loss = []
        self.w_list = []
        for _ in range(self.max_iter):
            dw = self.gradient(self.w)
            self.w = self.w - self.alpha * dw
            loss.append(np.mean((self._predict(self.x) - self.y)**2))
            self.w_list.append(self.w)
        return loss

    def predict(self, x):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        return self._predict(x)

    def _predict(self, x):
        return x.dot(self.w)

    def gradient(self, w):
        err = self._predict(self.x) - self.y
        dw = self.x.T.dot(err) / self.y.shape[0]
        return dw
