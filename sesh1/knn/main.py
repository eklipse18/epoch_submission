import numpy as np
from collections import Counter

data = np.array([
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],
    [110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana']
])

labels = ['Apple', 'Banana', 'Orange']

X = data[:, :3].astype(np.float64)

def min_max_normalize(X):
    return (X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0))

def z_score_normalize(X):
    return (X - np.mean(X, axis=0))/np.std(X, axis=0)

X = z_score_normalize(X)
y = np.array(list(map(lambda x: labels.index(x), data[:, 3])))

def euclid_d(x1, x2):
    return np.linalg.norm(x2-x1, 1)

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        return list(map(self.predict_one, X_test))

    def predict_one(self, x):
        i = list(range(len(self.X)))
        s = sorted(i, key=lambda t: euclid_d(x, self.X[t]))[:self.k]
        lbls = [self.y[t] for t in s]
        return Counter(lbls).most_common(1)[0][0]

    def accuracy(self, X_test) -> float:
        return np.mean(X_test[:, 3] == self.predict(X_test[:, :3]))

test_data = z_score_normalize(np.array([
    [118, 6.2, 0],  # Expected: Banana
    [160, 7.3, 1],  # Expected: Apple
    [185, 7.7, 2]   # Expected: Orange
]))
test_data2 = z_score_normalize(np.array([
    [118, 6.2, 0],
    [160, 7.3, 1],
    [185, 7.7, 2]
]))
test_data2 = np.hstack((test_data2, np.array([1, 0, 2]).reshape(3, 1)))

c = KNN()
c.fit(X, y)
print([labels[x] for x in c.predict(test_data)])
print(c.accuracy(test_data2))
