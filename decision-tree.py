from scipy.io import loadmat
import numpy as np
from math import log

def load_data(filename):
    """Returns a tuple of features and labels."""
    data = loadmat(filename, squeeze_me = True)
    return data['x'], data['y']

emotions = {'anger': 1,
            'disgust': 2,
            'fear': 3,
            'happiness': 4,
            'sadness': 5,
            'surprise': 6}

def map_label (labels, emotion):
    """Replace label with 1 or 0 if emotion is present or absent."""
    value = emotions.get(emotion, -1)
    return np.array([1 if lab == value else 0 for lab in labels])

def majority_value(binary_targets):
    count = 0
    for item in binary_targets:
        count += 1 if item == 1 else -1
    return 1 if count > 0 else 0

def entropy (pos, neg):
    if pos == 0 or neg == 0: return 0
    p = 1.0 * pos / (pos + neg)
    n = 1.0 * neg / (pos + neg)
    return - p * log(p, 2) - n * log(n, 2)

def remainder (attribute, sum_pn, examples, binary_targets):
    [p0, p1, n0, n1] = np.zeros(4, float)
    if sum_pn == 0:
        return 0
    for i in range(len(binary_targets)):
        if binary_targets[i] == 1:
            if examples[i, attribute] == 1:
                p1 += 1
            else:
                p0 += 1
        else:
            if examples[i, attribute] == 1:
                n1 += 1
            else:
                n0 += 1
    return (p0+n0)/sum_pn * entropy(p0, n0) + (p1+n1)/sum_pn * entropy(p1, n1)

def gain (attribute, examples, binary_targets):
    [p, n] = np.zeros(2, float)
    for item in binary_targets:
        if item == 1:
            p += 1
        else:
            n += 1
    return entropy(p, n) - remainder(attribute, p+n, examples, binary_targets)

def choose_best_attribute(examples, attributes, binary_targets):
    [best_gain, best] = np.zeros(2, int)
    for attribute in attributes:
        temp_gain = gain(attribute, examples, binary_targets)
        if temp_gain > best_gain:
            [best_gain, best] = [temp_gain, attribute]
    return best

class Node:

    def __init__ (self, kids, op = None, label = None):
        self.kids = kids
        self.op = op             # Attribute being tested
        self.label = label       # Classification at leaf nodes

    def add_kid (self, kid):
        self.kids.append(kid)

def find_elements(examples, binary_targets, attribute, value):
    index = []
    binary_targets_i = np.array([], dtype=int)
    for i in range(binary_targets.shape[0]):
        if examples[i, attribute] == value:
            index.append(i)
            binary_targets_i = np.append(binary_targets_i, binary_targets[i])
    return [examples[index, :], binary_targets_i]

def decision_tree_learning(examples, attributes, binary_targets):
    if len(set(binary_targets)) == 1:
#       print "leaf"
        return Node(kids = [], label = binary_targets[0])
    elif not attributes:
#       print "leaf"
        return Node(kids = [], label = majority_value(binary_targets))
    else:
        best_attribute = choose_best_attribute(examples, attributes, binary_targets)
#       print best_attribute
        tree = Node(kids = [], op = best_attribute)
        for i in range(2):
#           print "i =", i
            [examples_i, binary_targets_i] = find_elements(examples, binary_targets, best_attribute, i)
            if len(examples_i) == 0:
#               print "leaf"
                return Node(kids = [], label = majority_value(binary_targets))
            else:
                new_attribute = list(attributes)
                new_attribute.remove(best_attribute)
                subtree = decision_tree_learning(examples_i, new_attribute, binary_targets_i)
                tree.add_kid(subtree)
        return tree

def test_trees(T, features):
    while T.op != None:
        print T.op
        T = T.kids[features[T.op]]
    return bool(T.label)

def confusion_matrix(label_num, pre_act_class):
    # label number is the number of Classification
    # pre_act_class is a matrix that contain a column of predict Classification and a column of actual column
    resulut_matrix = np.zeros((label_num,label_num))
    for index in range(len(pre_act_class[0])):
        i = pre_act_class[1][index]
        j = pre_act_class[0][index]
        resulut_matrix[i][j] += 1
    return resulut_matrix


X, y = load_data("cleandata_students.mat")
attributes = list(xrange(45))

anger_targets     = map_label(y, "anger")
disgust_targets   = map_label(y, "disgust")
fear_targets      = map_label(y, "fear")
happiness_targets = map_label(y, "happiness")
sadness_targets   = map_label(y, "sadness")
surprise_targets  = map_label(y, "surprise")

anger_decision_tree = decision_tree_learning(X, attributes, anger_targets)
disgust_decision_tree = decision_tree_learning(X, attributes, disgust_targets)
fear_decision_tree = decision_tree_learning(X, attributes, fear_targets)
happiness_decision_tree = decision_tree_learning(X, attributes, happiness_targets)
sadness_decision_tree = decision_tree_learning(X, attributes, sadness_targets)
surprise_decision_tree = decision_tree_learning(X, attributes, surprise_targets)

for i in X:
print test_trees(sadness_decision_tree, X[1])
print X[1]
print y[1]

class1 = np.array([[0,1,2,0],[0,2,1,0]])
print confusion_matrix(3,class1)
