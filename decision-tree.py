from scipy.io import loadmat
import numpy as np
from math import log

def load_data(filename):
    data = loadmat(filename,squeeze_me=True)
    return data

def map_label(data, emotion):
    new_label = data.copy()
    for i in range(len(new_label)):
        new_label[i] = 1 if new_label[i] == get_label(emotion) else 0
    return new_label

def get_label(emotion):
    return {
        'anger': 1,
        'disgust': 2,
        'fear': 3,
        'happiness': 4,
        'sadness': 5,
        'surprise': 6
        }.get(emotion, -1)

def majority_value(binary_targets):
    count = 0
    for item in binary_targets:
       count += 1 if item == 1 else -1
    return 1 if count > 0 else 0

def entropy(p, n):
    sum_pn = float(p+n)
    first = 0.0 if p == 0 else - p/sum_pn * log(p/sum_pn, 2)
    second = 0.0 if n == 0 else - n/sum_pn * log(n/sum_pn, 2)
    return first + second

def remainder(attribute, sum_pn, examples, binary_targets):
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

def gain(attribute, examples, binary_targets):
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
    def __init__(self, kids, op=None, label=None):
        self.op = op
        self.kids = kids
        self.label = label

    def add_kid(self, kid):
        self.kids.append(kid)


def find_elements(examples, binary_targets, attribute, value):
    index = []
    binary_targets_i = np.array([], dtype=int)
    for i in range(binary_targets.shape[0]):
        if examples[i, attribute] == value:
            index.append(i)
            binary_targets_i = np.append(binary_targets_i, binary_targets[i])
    return [examples[index,:], binary_targets_i]

def decision_tree_learning(examples, attributes, binary_targets):
    if len(set(binary_targets)) == 1:
#        print "leaf"
        return Node(kids=[], label=binary_targets[0])
    elif not attributes:
#        print "leaf"
        return Node(kids=[], label=majority_value(binary_targets))
    else:
        best_attribute = choose_best_attribute(examples, attributes, binary_targets)
#        print best_attribute
        tree = Node(kids=[], op=best_attribute)
        for i in range(2):
#            print "i =", i
            [examples_i, binary_targets_i] = find_elements(examples, binary_targets, best_attribute, i)
            if len(examples_i) == 0:
#                print "leaf"
                return Node(kids=[], label=majority_value(binary_targets))
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


data = load_data("cleandata_students.mat")
attributes = list(xrange(45))

anger_targets = map_label(data['y'], "anger")
disgust_targets = map_label(data['y'], "disgust")
fear_targets = map_label(data['y'], "fear")
happiness_targets = map_label(data['y'], "happiness")
sadness_targets = map_label(data['y'], "sadness")
surprise_targets = map_label(data['y'], "surprise")

anger_decision_tree = decision_tree_learning(data['x'], attributes, anger_targets)
disgust_decision_tree = decision_tree_learning(data['x'], attributes, disgust_targets)
fear_decision_tree = decision_tree_learning(data['x'], attributes, fear_targets)
happiness_decision_tree = decision_tree_learning(data['x'], attributes, happiness_targets)
sadness_decision_tree = decision_tree_learning(data['x'], attributes, sadness_targets)
surprise_decision_tree = decision_tree_learning(data['x'], attributes, surprise_targets)

#for i in data['x']:
print test_trees(sadness_decision_tree, data['x'][1])
print data['x'][1]
print data['y'][1]
