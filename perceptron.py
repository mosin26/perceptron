import numpy as np


class Perceptron:
    def __init__(self, dim):
        self.w = np.random.rand(dim + 1) / 10**3

    def fit(self, x, y, n_epochs=1000):
        for epoch in range(n_epochs):
            for feature, label in zip(x, y):
                feature = np.concatenate(([1], feature))
                prediction = np.dot(self.w, feature.T)
                if prediction >= 0 and label < 0:
                    self.w -= feature
                elif prediction < 0 and label > 0:
                    self.w += feature

    def predict(self, x):
        y = []
        for feature in x:
            if np.dot(self.w, np.concatenate(([1], feature)).T) >= 0:
                y.append(1)
            else:
                y.append(-1)
        return np.array(y)
