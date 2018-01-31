from scipy.io import loadmat
import numpy as np
from math import log

def load_data(filename):
    """ Returns a tuple of features and labels """
    data = loadmat(filename, squeeze_me = True)
    return data['x'], data['y']

emotions = {'anger': 1,
            'disgust': 2,
            'fear': 3,
            'happiness': 4,
            'sadness': 5,
            'surprise': 6}

def binary_labels (labels, emotion):
    """ Replace label with 1 or 0 if emotion is present or absent """
    value = emotions.get(emotion, -1)
    return np.array([1 if lab == value else 0 for lab in labels])

class Node:

    def __init__ (self, op = None, kids = None, label = None):
        self.op = op             # Attribute being tested
        self.kids = kids if kids is not None else []
        self.label = label       # Classification at leaf nodes

class DecisionTree:

    def __init__ (self, max_depth = None):
        self.max_depth = max_depth
        self.nodes = []          # Root node is nodes[0]

    def fit (self, X, y):
        """ y must be a vector of 0s and 1s """
        if all(lab == y[0] for lab in y):
            nodes.append(Node(label = y[0]))
        return self

def majority_value(binary_targets):
    sum = 0
    for item in binary_targets:
        sum += 1 if item == 1 else -1
    return 1 if sum > 0 else 0

def entropy (pos, neg):
    if pos == 0 or neg == 0: return 0
    p = 1.0 * pos / (pos + neg)
    n = 1.0 * neg / (pos + neg)
    return - p * log(p, 2) - n * log(n, 2)

def remainder(attribute, p, n, data):
    [p0, p1, n0, n1] = np.zeros(4, float)
    sum = p+n
    if sum == 0:
        return 0
    for i in range(len(y)):
        if y[i] == 1:
            if X[i, attribute] == 1:
                p1 += 1
            else:
                p0 += 1
        else:
            if X[i, attribute] == 1:
                n1 += 1
            else:
                n0 += 1
    return (p0+n0)/sum * entropy(p0, n0) + (p1+n1)/sum * entropy(p1, n1)

def gain(attribute, data):
    [p, n] = np.zeros(2, float)
    for item in y:
        if item == 1:
            p += 1
        else:
            n += 1
    return entropy(p, n) - remainder(attribute, p, n, data)

X, y = load_data("cleandata_students.mat")
y = binary_labels(y, "happiness")
