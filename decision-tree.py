from scipy.io import loadmat
import numpy as np
from math import log

"""
Load data from a Matlab file
The data contains a tuple of features and labels

@param filename   - file's path
@return           - tuple of (features, labels)
"""
def load_data(filename):
    data = loadmat(filename, squeeze_me = True)
    return data['x'], data['y']

emotions = {'anger'     : 1,
            'disgust'   : 2,
            'fear'      : 3,
            'happiness' : 4,
            'sadness'   : 5,
            'surprise'  : 6}

"""
Replace label with 1 or 0 if emotion is present or absent respectively

@param labels    - list of labels from 1 to 6
@param emotion   - the emotion to be classified
@return          - list of labels 1 or 0 if emotion is present or absent
"""
def map_label (labels, emotion):
    value = emotions.get(emotion, -1)
    return np.array([1 if lab == value else 0 for lab in labels])

"""
Find the majority value of a binary list

@param binary_targets - list with values 0 and 1
@return               - 0 if the number of 0's is more than that of 1's
                        1 if the number of 1's is more than that of 0's
"""
def majority_value(binary_targets):
    count = 0
    for item in binary_targets:
        count += 1 if item == 1 else -1
    return 1 if count > 0 else 0

"""
Compute entropy

@param pos    - number of positive examples
@param neg    - number of negative examples
@return       - entropy
"""
def entropy (pos, neg):
    if pos == 0 or neg == 0: return 0
    p = 1.0 * pos / (pos + neg)
    n = 1.0 * neg / (pos + neg)
    return - p * log(p, 2) - n * log(n, 2)

"""

@param examples          - NumPy array with shape (N, P)
@param binary_targets    - NumPy array of 0's and 1's with length N
"""
"""
Recommended change:

    p0 = np.sum(examples[:, attribute] == 0 && binary_targets == 1)
    p1 = np.sum(examples[:, attribute] == 1 && binary_targets == 1)
    n0 = np.sum(examples[:, attribute] == 0 && binary_targets == 0)
    n1 = np.sum(examples[:, attribute] == 1 && binary_targets == 0)
    return (p0 + n0) / sum_pn * entropy(p0, n0) + (p1 + n1) / sum_pn * entropy(p1, n1
    )
"""
def remainder (attribute, sum_pn, examples, binary_targets):
    if sum_pn == 0: return 0
    p0 = p1 = n0 = n1 = 0.0
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
    p = n = 0.0
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
    binary_targets_i = np.array([], dtype = int)
    for i in range(binary_targets.shape[0]):
        if examples[i, attribute] == value:
            index.append(i)
            binary_targets_i = np.append(binary_targets_i, binary_targets[i])
    return [examples[index, :], binary_targets_i]

"""
"""
def decision_tree_learning(examples, attributes, binary_targets):
    if len(set(binary_targets)) == 1:
        #print "leaf"
        return Node(kids = [], label = binary_targets[0])
    elif not attributes:
        #print "leaf"
        return Node(kids = [], label = majority_value(binary_targets))
    else:
        best_attribute = choose_best_attribute(examples, attributes, binary_targets)
        #print best_attribute
        tree = Node(kids = [], op = best_attribute)
        for i in range(2):
            #print "i =", i
            [examples_i, binary_targets_i] = find_elements(examples, binary_targets, best_attribute, i)
            if len(examples_i) == 0:
                #print "leaf"
                return Node(kids = [], label = majority_value(binary_targets))
            else:
                new_attribute = list(attributes)
                new_attribute.remove(best_attribute)
                subtree = decision_tree_learning(examples_i, new_attribute, binary_targets_i)
                tree.add_kid(subtree)
    return tree

"""
"""
def test_single_tree(tree, features):
    depth = 0
    while tree.op != None:
        tree = tree.kids[features[tree.op]]
        depth += 1
    return tree.label, depth

"""
"""
def highest_score(scores):
    return scores.index(max(scores)) + 1

"""
"""
def most_similar(T, features, labels):
    if sum(labels) <= 1:
        return labels.index(sum(labels)) + 1
    scores = np.zeros(len(emotions)).tolist()
    for i in range(len(features)):
        if features[i] == 1:
            features[i] = 0
            for j in range(len(labels)):
                if labels[j] == 1:
                    new_label, depth = test_single_tree(T[j], features)
                    scores[j] += new_label
            features[i] = 1
    return highest_score(scores)

"""
"""
def testTrees(T, x2):
    predictions = []
    for x in x2:
        scores, labels = [], []
        for tree in T:
            label, depth = test_single_tree(tree, x)
            labels.append(label)
            if label == 0: depth = 0
            scores.append(depth)
        # plan A
        predictions.append(highest_score(labels))
        
        # plan B
        #predictions.append(highest_score(scores))

        # plan C
        #predictions.append(most_similar(T, x, labels))

    return np.array(predictions)


"""
Compute a confusion matrix

@param pre_act_class    - Matrix with 2 rows:
                          First row is the predict class for examples;
                          Second row is the actual Classification for examples;
@param label_num        - number of classifications
@return                 - confusion matrix
"""
def confusion_matrix (label_num, pre_act_class):
    confusion = np.zeros((label_num,label_num))
    for index in range(len(pre_act_class[0])):
        i = pre_act_class[1][index] - 1
        j = pre_act_class[0][index] - 1
        confusion[i][j] += 1
    return confusion

def recall_precision_rates(label_num, confusion_matrix):
    res_rec_prec = []
    for index in range(label_num):
        recall_rate = get_recall_rate(confusion_matrix, index)
        predict_rate = get_predict_rate(confusion_matrix, index)
        res_rec_prec.append([recall_rate, predict_rate])
    return res_rec_prec

def get_recall_rate(confusion_matrix, index):
    tp = confusion_matrix[index,index]
    recall_rate = 1.0 * tp /sum(confusion_matrix[index])
    return recall_rate

def get_predict_rate(confusion_matrix, index):
    tp = confusion_matrix[index,index]
    predict_rate = 1.0 * tp / sum(confusion_matrix[:,index])
    return predict_rate

def fa_measure(a, label_num, res_rec_prec):
    meas_rel = []
    for index in range(label_num):
        recall = res_rec_prec[index][0]
        precision = res_rec_prec[index][1]
        fa_i = (1.0 + a**2)* (precision * recall) / (a * precision + recall)
        meas_rel.append(fa_i)
    return meas_rel

def classfi_rate(label_num, confusion_matrix):
    dig_mat = np.diag(np.ones(label_num))
    correct_class_num = sum(map(sum, dig_mat * confusion_matrix))
    total_num = sum(map(sum, confusion_matrix))
    return correct_class_num / total_num


def n_fold(data, labels, n):
    length = len(data) / n
    avg_classfi_rate = 0
    for i in range(n):
        # the ith time
        # divide the data into training_data, valification_data and testing_data
        testing_data_index = [x + i*length for x in range(length)]
        training_data_index = [index for index in range(len(data)) if index not in testing_data_index]
        testing_data = data[testing_data_index]
        result_test = [labels[i] for i in testing_data_index]
        training_data = data[training_data_index]
        binary_targets_train = [labels[i] for i in training_data_index]
        # trai&val_data_index = [index for index in range(len(data)) if index not in testing_data_index]
        # trai&val_data = data[trai&val_data_index]
        # valification_data = trai&val_data[0:length]
        # _data = trai&val_data[length:]
        # using training_data to train tress

        targets = []
        for j in range(len(emotions)):
            e = emotions.keys()[emotions.values().index(j+1)]
            targets.append(map_label(binary_targets_train, e))

        attributes = range(len(data[0]))

        T = []
        for t in targets:
            T.append(decision_tree_learning(training_data, attributes, t))

        # calculate the precision rate
        #predictx =[]
        #T = [anger_decision_tree, disgust_decision_tree, fear_decision_tree,
        #        happiness_decision_tree, sadness_decision_tree, surprise_decision_tree]
        #for data in testing_data:
        predictx = testTrees(T, testing_data)

        con_mat = confusion_matrix(len(emotions), [predictx.tolist(), result_test])
        avg_classfi_rate += classfi_rate(len(emotions), con_mat)
    return avg_classfi_rate / n


# --- Test ---

X, y = load_data("cleandata_students.mat")
nx, ny = load_data("noisydata_students.mat")
attributes = list(xrange(45))

print n_fold(X, y, 10)
"""
anger_targets      = map_label(y[0:len(X)*9/10], "anger")
disgust_targets    = map_label(y[0:len(X)*9/10], "disgust")
fear_targets       = map_label(y[0:len(X)*9/10], "fear")
happiness_targets  = map_label(y, "happiness")
sadness_targets    = map_label(y[0:len(X)*9/10], "sadness")
surprise_targets   = map_label(y[0:len(X)*9/10], "surprise")

test = map_label(ny,"happiness")
td = X[len(X)*9/10:len(X),:]
vd = X[0:len(X)*9/10,:]

anger_decision_tree     = decision_tree_learning(vd, attributes, anger_targets)
disgust_decision_tree   = decision_tree_learning(vd, attributes, disgust_targets)
fear_decision_tree      = decision_tree_learning(vd, attributes, fear_targets)
happiness_decision_tree = decision_tree_learning(vd, attributes, happiness_targets)
sadness_decision_tree   = decision_tree_learning(vd, attributes, sadness_targets)
surprise_decision_tree  = decision_tree_learning(vd, attributes, surprise_targets)
"""
# calculate the precision rate
# predictx =[]
# T = [anger_decision_tree, disgust_decision_tree, fear_decision_tree,
#         happiness_decision_tree, sadness_decision_tree, surprise_decision_tree]
# for i in td:
#     predictx.append(testTrees(T, i))
#     """## TODO : fix bug
#     if(classify_emotion(i)==None):
#         predictx.append(0)
#     else:
#         predictx.append(classify_emotion(i))
#     """
# diff = (predictx - y[len(X)*9/10:len(X)])
# print float(sum(x == 0 for x in diff))/len(diff)
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
