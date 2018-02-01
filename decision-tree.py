from scipy.io import loadmat
import numpy as np
from math import log

"""
# function to load data from a matlab file
# the data contains a tuple of features and labels
#
# @param filename   - the path of the file
# @return           - data of features, and data of labels
"""
def load_data(filename):
    data = loadmat(filename, squeeze_me = True)

    return data['x'], data['y']

#map emotions to labels
emotions = {'anger'     : 1,
            'disgust'   : 2,
            'fear'      : 3,
            'happiness' : 4,
            'sadness'   : 5,
            'surprise'  : 6}

"""
# function to replace label with 1 or 0 for emotion present or absent respectively
#
# @param labels     - a list of labels from 1 to 6
# @param emotion    - the emotion to classify
# @return           - a list of labels 1 or 0 for emotion present of absent
"""
def map_label (labels, emotion):
    value = emotions.get(emotion, -1)

    return np.array([1 if lab == value else 0 for lab in labels])

"""
# function to compute the majority value of a list
#
# @param binary_targets - a list with values 0 and 1
# @return               - 0 if the number of 0 is more than that of 1
                          1 if the number of 1 is more than that of 0
"""
def majority_value(binary_targets):
    count = 0
    for item in binary_targets:
        count += 1 if item == 1 else -1

    return 1 if count > 0 else 0

"""
# function to compute the entropy
#
# @param pos    - number of positive examples
# @param neg    - number of negative examples
# @return       - the entropy
"""
def entropy (pos, neg):
    if pos == 0 or neg == 0: return 0
    p = 1.0 * pos / (pos + neg)
    n = 1.0 * neg / (pos + neg)

    return - p * log(p, 2) - n * log(n, 2)

"""
"""
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

"""
"""
def gain (attribute, examples, binary_targets):
    [p, n] = np.zeros(2, float)
    for item in binary_targets:
        if item == 1:
            p += 1
        else:
            n += 1
    return entropy(p, n) - remainder(attribute, p+n, examples, binary_targets)

"""
"""
def choose_best_attribute(examples, attributes, binary_targets):
    [best_gain, best] = np.zeros(2, int)
    for attribute in attributes:
        temp_gain = gain(attribute, examples, binary_targets)
        if temp_gain > best_gain:
            [best_gain, best] = [temp_gain, attribute]
    return best

"""
"""
class Node:

    def __init__ (self, kids, op = None, label = None):
        self.kids = kids
        self.op = op             # Attribute being tested
        self.label = label       # Classification at leaf nodes

    def add_kid (self, kid):
        self.kids.append(kid)

"""
"""
def find_elements(examples, binary_targets, attribute, value):
    index = []
    binary_targets_i = np.array([], dtype=int)
    for i in range(binary_targets.shape[0]):
        if examples[i, attribute] == value:
            index.append(i)
            binary_targets_i = np.append(binary_targets_i, binary_targets[i])

    return [examples[index, :], binary_targets_i]

"""
"""
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

"""
"""
def test_trees(T, features):
    while T.op != None:
        T = T.kids[features[T.op]]
    return T.label

"""
"""
def classify_emotion(examples):
    result = [test_trees(anger_decision_tree, examples),
              test_trees(disgust_decision_tree, examples),
              test_trees(fear_decision_tree, examples),
              test_trees(happiness_decision_tree, examples),
              test_trees(sadness_decision_tree, examples),
              test_trees(surprise_decision_tree, examples)]
    for i in range(6):
        if result[i] == 1:
            return i+1
    
    return 1


"""
# function to compute a confusion matrix
#
# @param pre_act_class    - a matrix contains two rows:
                            the first row is the predict class for examples;
                            the second row is the actual Classification for examples;
# @param label_num        - numbers of classification
# @return                 - confusion matrix
"""
def confusion_matrix(label_num, pre_act_class):
    resulut_matrix = np.zeros((label_num,label_num))
    for index in range(len(pre_act_class[0])):
        i = pre_act_class[1][index]
        j = pre_act_class[0][index]
        resulut_matrix[i][j] += 1
    return resulut_matrix

def recall_precision_rates(label_num, confusion_matrix):
    res_rec_prec = []
    for index in range(label_num):
        recall_rate = get_recall_rate(confusion_matrix, index)
        predict_rate = get_predict_rate(confusion_matrix, index)
        res_rec_prec.append([recall_rate,predict_rate])
    return res_rec_prec

def get_recall_rate(confusion_matrix, index):
    tp = confusion_matrix[index,index]
    recall_rate = float(tp)/sum(confusion_matrix[index])
    return recall_rate
def get_predict_rate(confusion_matrix, index):
    tp = confusion_matrix[index,index]
    predict_rate = float(tp)/sum(confusion_matrix[:,index])
    return predict_rate


def fa_measure(a, label_num, res_rec_prec):
    meas_rel = []
    for index in range(label_num):
        recall = res_rec_prec[index][0]
        precision = res_rec_prec[index][1]
        fa_i = (1.0+a)* (precision * recall) / (a * precision + recall)
        meas_rel.append(fa_i)
    return meas_rel

def ave_classfi_rate(label_num, confusion_matrix):
    dig_mat = np.diag(np.ones(label_num))
    correct_class_num = sum(map(sum,dig_mat * confusion_matrix))
    total_num = sum(map(sum,confusion_matrix))
    return correct_class_num/ total_num




"""
---- test ----

"""
X, y = load_data("cleandata_students.mat")
nx, ny = load_data("noisydata_students.mat")
attributes = list(xrange(45))

anger_targets       = map_label(y, "anger")
disgust_targets     = map_label(y, "disgust")
fear_targets        = map_label(y, "fear")
happiness_targets   = map_label(y, "happiness")
sadness_targets     = map_label(y, "sadness")
surprise_targets    = map_label(y, "surprise")

test = map_label(ny,"sadness")
#
# anger_decision_tree     = decision_tree_learning(X, attributes, anger_targets)
# disgust_decision_tree   = decision_tree_learning(X, attributes, disgust_targets)
# fear_decision_tree      = decision_tree_learning(X, attributes, fear_targets)
# happiness_decision_tree = decision_tree_learning(X, attributes, happiness_targets)
# sadness_decision_tree   = decision_tree_learning(X, attributes, sadness_targets)
# surprise_decision_tree  = decision_tree_learning(X, attributes, surprise_targets)

# calculate the precision rate
predictx =[]
for i in nx:
    predictx.append(classify_emotion(i))
    """## TODO : fix bug
    if(classify_emotion(i)==None):
        predictx.append(0)
    else:
        predictx.append(classify_emotion(i))
    """
diff = (predictx - ny)
print float(sum(x == 0 for x in diff))/len(diff)
# predictx =[]
# for i in nx:
#     ## TODO : fix bug
#     if(classify_emotion(i)==None):
#         predictx.append(0)
#     else:
#         predictx.append(classify_emotion(i))
# diff = (predictx - ny)
# print float(sum(x == 0 for x in diff))/len(diff)
# print classify_emotion(nx[342]),ny[342]
#for i in X:
#print test_trees(sadness_decision_tree, X[1])

#class1 = np.array([[0,1,2,0],[0,2,1,0]])
#print confusion_matrix(3,class1)
