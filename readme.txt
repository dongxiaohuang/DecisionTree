The six decision trees are stored in the file 'Trees.pkl'. 
To load the trees, read the file in python as following:

import pickle
import decision_tree
filename = open('Trees.pkl', 'rb')
T = pickle.load(filename)

where T will be a list of six tree roots.
To use the trees, invoke the method testTrees() as:

predictions = testTrees(T, x2)

where x2 should be two-dimensional numPy array of examples, and return value will also be one-dimension numPy array of labels.
