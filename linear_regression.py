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
        self.x = x.reshape(y.shape[0], -1)
        self.y = y

        self.w = np.random.normal(1, 0.1, (x.shape[1],))
        self.b = np.random.normal(1, 0.1)
        loss = []
        for _ in range(self.max_iter):
            dw, db = self.gradient(self.w, self.b)
            self.w = self.w - self.alpha * dw
            self.b = self.b - self.alpha * db
            loss.append(np.mean((self.predict(self.x) - self.y)**2))
        return loss

    def predict(self, x):
        return x.dot(self.w) + self.b

    def gradient(self, w, b):
        dw = self.x.T.dot(self.x.dot(w) + b - self.y) / self.y.shape[0]
        db = np.mean(self.x.dot(w) + b - self.y)
        return dw, db
