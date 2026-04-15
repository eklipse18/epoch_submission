import numpy as np

data = [
    [12.0, 1.5, 1, 'Wine'],
    [5.0, 2.0, 0, 'Beer'],
    [40.0, 0.0, 1, 'Whiskey'],
    [13.5, 1.2, 1, 'Wine'],
    [4.5, 1.8, 0, 'Beer'],
    [38.0, 0.1, 1, 'Whiskey'],
    [11.5, 1.7, 1, 'Wine'],
    [5.5, 2.3, 0, 'Beer']
]

labels = ['Wine', 'Beer', 'Whiskey']
X = np.array([x[:3] for x in data])
y = np.array([labels.index(x[3]) for x in data])

def normalize(X: np.ndarray) -> np.ndarray:
    return (X - np.mean(X, axis=0))/np.std(X, axis=0)

X = normalize(X)

print(X, y)

def gini_impurity(X: np.ndarray, y: np.ndarray):
    i = 1
    for k in range(len(labels)):
        i -= ((y==k).sum()/len(y))**2
    return i

def gini_gain(X: np.ndarray, y: np.ndarray, feature: int, threshold: float):  # Binary split only
    i1 = X[:, feature] >= threshold
    sx1, sx2 = X[i1], X[~i1]
    sy1, sy2 = y[i1], y[~i1]
    return gini_impurity(X, y) - np.average([gini_impurity(sx1, sy1), gini_impurity(sx2, sy2)], weights=[i1.sum(), (~i1).sum()])

def entropy(X: np.ndarray, y: np.ndarray):
    p = np.array([(y==k).sum()/len(X) for k in range(len(labels))])
    p = p[p>0]
    return -(p*np.log2(p)).sum()

def inf_gain(X: np.ndarray, y: np.ndarray, feature: int, threshold: float):  # Binary split only
    e = entropy(X, y)
    i1 = X[:, feature] >= threshold
    s1, s2 = y[i1], y[~i1]    
    return e - np.average([entropy(X[i1], y[i1]), entropy(X[~i1], y[~i1])], weights=[i1.sum(), (~i1).sum()])

print(gini_impurity(X, y))

class Node:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.leaf = False
        self.feature_index = None
        self.threshold = None
        self.left: 'Node | None' = None
        self.right: 'Node | None' = None
        if len(set(y)) == 1:
            self.value = y[0]
            print(self.value)
            self.leaf = True
        else:
            self.value = None

    def best_split(self):
        if self.leaf: # We've already determined this to be a leaf node, no need for splitting
            return
        i_best = -1
        t_best = -1
        g_best = -np.inf
        for i in range(len(labels)):
            vals = np.sort(np.unique(self.X[:, i]))
            ts = (vals[:-1] + vals[1:])/2
            for t in ts:
                g = gini_gain(self.X, self.y, i, t)  # swap out to inf_gain to use entropy
                if g >= g_best:
                    i_best = i
                    t_best = t
                    g_best = g
                
        if i_best != -1:
            self.feature_index = i_best
            self.threshold = t_best
            i1 = self.X[:, i_best] >= t_best
            i2 = ~i1
            self.left = Node(self.X[i1], self.y[i1])
            self.right = Node(self.X[i2], self.y[i2])
        
            self.left.best_split()
            self.right.best_split()

    def predict(self, data: np.ndarray) -> int:
        if self.leaf: 
            if self.value is None: # gotta do this to stop pyright from screaming
                raise Exception("empty leaf node")
            return self.value
        if not (self.right and self.left):
            raise Exception("Uninitialized node")
        if data[self.feature_index] >= self.threshold:
            return self.left.predict(data)
        else:
            return self.right.predict(data)

    def __str__(self) -> str:
        if self.leaf:
            return f"{{LNODE ({self.value})}}"
        else:
            return f"{{DNODE ({self.threshold} {self.feature_index}): {{{self.left}, {self.right}}}"

class Tree:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.root = Node(X, y)

    def initialize(self):
        self.root.best_split()

    def predict(self, data: np.ndarray) -> int:
        return self.root.predict(data)

    def multi_predict(self, X: np.ndarray) -> list[int]:
        return [self.root.predict(x) for x in X]

    def __str__(self) -> str:
        return f"Root: {self.root}"

test_data = normalize(np.array([
    [6.0, 2.1, 0],   # Expected: Beer
    [39.0, 0.05, 1], # Expected: Whiskey
    [13.0, 1.3, 1]   # Expected: Wine
]))

tree = Tree(X, y)
tree.initialize()

print(tree)
print([labels[i] for i in tree.multi_predict(test_data)])
