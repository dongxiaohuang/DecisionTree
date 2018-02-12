The six decision trees are stored in the file 'Trees.pkl'. 

To load the trees, read the file in python 2.7 as following:
```
import pickle
import decision_tree as dt
filename = open('Trees.pkl', 'rb')
T = pickle.load(filename)
```
where T will be a list of six tree roots.

To use the trees, invoke the method testTrees() as:

`predictions = dt.testTrees(T, x2)`

where x2 should be two-dimensional numPy array of examples, and return value will also be one-dimensional numPy array of labels.

To visuallize the trees, install graphviz first by 

`pip install graphviz`

`import graphviz`