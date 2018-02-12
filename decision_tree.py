from scipy.io import loadmat
import numpy as np
from math import log
import pickle
from visualization import treeviz, build_tree

"""
Load data from a Matlab file

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
    n = 1.0 - p
    return - p * log(p, 2) - n * log(n, 2)

"""
Compute information gain of an attribute

@param attribute        - Integer
@param examples         - NumPy array with shape (N, P)
@param binary_targets   - NumPy array of 0's and 1's with length N
"""
def gain (attribute, examples, binary_targets):
    p = np.sum(binary_targets == 1)
    n = np.sum(binary_targets == 0)
    p0 = np.sum((examples[:, attribute] == 0) & (binary_targets == 1))
    p1 = np.sum((examples[:, attribute] == 1) & (binary_targets == 1))
    n0 = np.sum((examples[:, attribute] == 0) & (binary_targets == 0))
    n1 = np.sum((examples[:, attribute] == 1) & (binary_targets == 0))
    return entropy(p, n) - (p0 + n0) / float(p + n) * entropy(p0, n0) - (p1 + n1) / float(p + n) * entropy(p1, n1)

"""
Choose the attribute with highest information gain

@param examples         - list of data (AU)
@param attributes       - available attributes to choose
@param binary_targets   - list of labels with 0 and 1
@return                 - index of attribute with highest gain
"""
def choose_best_attribute(examples, attributes, binary_targets):
    best_gain, best = 0, attributes[0]
    for attribute in attributes:
        temp_gain = gain(attribute, examples, binary_targets)
        if temp_gain > best_gain:
            best_gain, best = temp_gain, attribute
    return best

"""
A node in a tree, which contains
kids    - subtrees
op      - attribute being tested
label   - classification at leaf nodes
"""
class Node:

    def __init__ (self, kids, op = None, label = None):
        self.kids = kids
        self.op = op
        self.label = label

    # Add a subtree to kids
    def add_kid (self, kid):
        self.kids.append(kid)

"""
Find subset of targets for each attribute equal to a certain value

@param examples         - list of data (AU)
@param binary_targets   - list of labels with 0 and 1
@param attribute        - the attribute being checked
@param value            - value of 0 or 1
@return                 - a subset of targets whose attribute would be all 0 or 1
"""
def find_elements(examples, binary_targets, attribute, value):
    index = []
    binary_targets_i = np.array([], dtype = int)
    for i in range(binary_targets.shape[0]):
        if examples[i, attribute] == value:
            index.append(i)
            binary_targets_i = np.append(binary_targets_i, binary_targets[i])
    return examples[index, :], binary_targets_i

"""
Construct a decision tree

@param examples         - list of data (AU)
@param attributes       - available attributes to choose one as the best attribute
@param binary_targets   - list of labels with 0 and 1
@return                 - a node of the tree, root, internal node, or leaf node
"""
def decision_tree_learning(examples, attributes, binary_targets, depth = 0):
    if len(set(binary_targets)) == 1:
        # print "leaf"
        return Node(kids = [], label = binary_targets[0])
    elif not attributes or depth > 100:
        # print "leaf"
        return Node(kids = [], label = majority_value(binary_targets))
    else:
        best_attribute = choose_best_attribute(examples, attributes, binary_targets)
        # print best_attribute
        tree = Node(kids = [], op = best_attribute)
        for i in range(2):
            #print "i =", i
            examples_i, binary_targets_i = find_elements(examples, binary_targets, best_attribute, i)
            if len(examples_i) == 0:
                #print "leaf"
                return Node(kids = [], label = majority_value(binary_targets))
            else:
                new_attribute = list(attributes)
                new_attribute.remove(best_attribute)
                subtree = decision_tree_learning(examples_i, new_attribute, binary_targets_i, depth+1)
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
def most_similar(T, features):
    #if sum(labels) == 1:
        #return labels.index(sum(labels)) + 1
    scores = np.zeros(len(emotions), int).tolist()
    for i in range(len(features)):
        #features[i] = (features[i] - 1) ** 2
        #for k in range(i+1, len(features)):
        #if features[i] == 1:
        features[i] = 0 if features[i] == 1 else 1
        for j in range(len(emotions)):
                #if labels[j] == 1:
            new_label, depth = test_single_tree(T[j], features)
            scores[j] += new_label
        features[i] = 0 if features[i] == 1 else 1
        #features[i] = (features[i] - 1) ** 2
    print "scores: ", scores
    return highest_score(scores)

"""
"""
def testTrees(T, x2, y):
    predictions = []
    i = -1
    for x in x2:
        i += 1
        scores, labels = [], []
        for tree in T:
            label, depth = test_single_tree(tree, x)
            labels.append(label)
            #if label == 0: depth = 0
            scores.append(depth)
        # plan A
        #predictions.append(highest_score(labels))

        # plan B
        #if sum(labels) > 1:
        #predictions.append(highest_score(scores))

        # plan C
        #if sum(labels) <= 1:
        print "------"
        print "labels: ", labels
        #print "depth: ", scores
        print "true: ", y[i]
        predictions.append(most_similar(T, x))

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
        actual_i = pre_act_class[1][index] - 1
        predict_j = pre_act_class[0][index] - 1
        confusion[actual_i][predict_j] += 1
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

"""
"""
def generate_trees(training_data, binary_targets):
    targets = []
    for j in range(len(emotions)):
        e = emotions.keys()[emotions.values().index(j+1)]
        targets.append(map_label(binary_targets, e))
    attributes = range(len(training_data[0]))
    T = []
    for t in targets:
        T.append(decision_tree_learning(training_data, attributes, t, 0))
    return T

"""
"""
def n_fold(data, labels, n):
    length = len(data) / n
    avg_classfi_rate = 0
    result_test = []
    predictx = []
    for i in range(n):
        # the ith time
        # divide the data into training_data, valification_data and testing_data
        testing_data_index = [x + i*length for x in range(length)]
        training_data_index = [index for index in range(len(data)) if index not in testing_data_index]
        testing_data = data[testing_data_index]

        result_test += [labels[i] for i in testing_data_index]
        te = [labels[i] for i in testing_data_index]

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
        predictx += testTrees(T, testing_data, te).tolist()

    con_mat = confusion_matrix(len(emotions), [predictx, result_test])
    avg_classfi_rate += classfi_rate(len(emotions), con_mat)
    
    return avg_classfi_rate


# --- Test ---

x, y = load_data("cleandata_students.mat")
nx, ny = load_data("noisydata_students.mat")

#T = generate_trees(x, y)

#for index in range(len(T)):
#    tree_graph = build_tree(T[index])
#    treeName = e = emotions.keys()[emotions.values().index(index+1)]+'Tree'
#    g = treeviz(tree_graph, tree_name = treeName)
#    g.render(treeName,view=True)

#output = open('Trees.pkl', 'wb')
#pickle.dump(T, output)
#output.close()

#f = open('Trees.pkl', 'rb')
#T = pickle.load(f)
#print testTrees(T, nx)

print n_fold(x, y, 10)
