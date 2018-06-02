import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def _insert_bias(x):
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    return x


class LogisticRegression(object):
    def __init__(self, alpha, max_iter, lamd=0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.lamd = lamd

    def fit(self, x, y, intercept_adapt=True):
        if intercept_adapt:
            x = _insert_bias(x)
        self.x = x
        self.y = y

        self.w = np.random.normal(1, 0.1, (self.x.shape[1],))
        self.loss = []
        self.w_list = []
        for _ in range(self.max_iter):
            dw = self.gradient(self.w)
            self.w -= self.alpha * dw
            self.w_list.append(self.w)
            self.loss.append(self.cost(self.x, self.y, False))
        return self.loss, self.w_list

    def predict(self, x, intercept_adapt=True):
        if intercept_adapt:
            x = _insert_bias(x)
        return sigmoid(x.dot(self.w))

    def gradient(self, w):
        err = self.predict(self.x, False) - self.y
        dw = self.x.T.dot(err) / self.y.shape[0]
        dw[1:] += self.lamd / self.y.shape[0] * w[1:]
        return dw

    def cost(self, x, y, intercept_adapt=True):
        p = self.predict(x, intercept_adapt)
        h = y * np.log(p) + (1-y) * np.log(1-p)
        return -np.mean(h) + self.lamd/2.0/y.shape[0]*np.sum(self.w[1:]**2)


class MutiLogisticRegression(LogisticRegression):
    def __init__(self, k, *args, **kwargs):
        self.k = k
        super(MutiLogisticRegression, self).__init__(*args, **kwargs)

    def fit(self, x, y, intercept_adapt=True):
        if intercept_adapt:
            x = _insert_bias(x)
        self.x = x
        self.y = y
        self.w = np.random.normal(1, 0.1, size=(self.k, self.x.shape[1]))
        self.loss = []
        self.w_list = []
        for _ in range(self.max_iter):
            dw = self.gradient(self.w)
            self.w -= self.alpha * dw
            self.w_list.append(self.w)
            self.loss.append(self.cost(self.x, self.y, False))
        return self.loss, self.w_list

    def gradient(self, w):
        err = self.predict(self.x, False) - self.y
        dw = err.T.dot(self.x) / self.y.shape[0]
        dw[1:] += self.lamd / self.y.shape[0] * w[1:]
        return dw

    def predict(self, x, intercept_adapt=True):
        if intercept_adapt:
            x = _insert_bias(x)
        return sigmoid(x.dot(self.w.T))

    def cost(self, x, y, intercept_adapt=True):
        p = self.predict(x, intercept_adapt)
        h = y * np.log(p) + (1-y) * np.log(1-p)
        return -np.mean(h, axis=0) + self.lamd/self.y.shape[0]*np.sum(self.w**2, axis=1)

