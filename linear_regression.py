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
        self.x = x
        self.y = y
        self.w, self.b = np.random.normal(1, 0.1), np.random.normal(1, 0.1)
        loss = []
        for _ in range(self.max_iter):
            dw, db = self.gradient(self.w, self.b)
            self.w = self.w - self.alpha * dw
            self.b = self.b - self.alpha * db
            loss.append(np.mean((self.f(self.x) - self.y)**2))
        return loss

    def f(self, x):
        return self.w * x + self.b

    def predict(self, x):
        return self.f(x)

    def gradient(self, w, b):
        dw = np.mean(self.x * (w * self.x + b - self.y))
        db = np.mean(w * self.x + b - self.y)
        return dw, db
