import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class LogisticRegression(object):
    def __init__(self, alpha, max_iter):
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        self.x = np.hstack((np.ones((x.shape[0], 1)), x))
        self.y = y

        self.w = np.random.normal(1, 0.1, (self.x.shape[1],))
        loss = []
        for _ in range(self.max_iter):
            dw = self.gradient(self.w)
            self.w = self.w - self.alpha * dw

            h = self._predict(self.x)
            loss.append(-np.mean(self.y * np.log(h) + (1 - self.y) * np.log(1 - h)))
        return loss

    def predict(self, x):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        return self._predict(x)

    def _predict(self, x):
        return sigmoid(x.dot(self.w))

    def gradient(self, w):
        err = self._predict(self.x) - self.y
        dw = self.x.T.dot(err) / self.y.shape[0]
        return dw
