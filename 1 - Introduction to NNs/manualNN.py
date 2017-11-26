'''TensorFlow structure for classification problem'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
#%matplotlib inline #uncomment if using Jupyter Notebook


class Operation():
    '''Computational operation wrapper'''

    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)

        _default_graph.operations.append(self)

    def compute(self, x, *y):
        '''Produces the output of the computation of this operation on parameters'''
        # to be overwritten by extending classes
        pass


class Add(Operation):
    '''Operation subclass, performs sum of two values'''

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        # override compute method from Operation class
        self.inputs = [x, y]
        return x + y  # perform sum


class Multiply(Operation):
    '''Operation subclass, performs multiplication of two values'''

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        # override compute method from Operation class
        self.inputs = [x, y]
        return x * y  # perform scalar multiplication


class Matmul(Operation):
    '''Operation subclass, performs dot multiplication of two matrices'''

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        # override compute method from Operation class
        self.inputs = [x, y]
        return x.dot(y)  # perform matrix multiplication


class Placehoder():
    '''Empty node that needs data to be provided to compute output'''

    def __init__(self):
        self.output_nodes = []
        _default_graph.placehoders.append(self)


class Variable():
    '''Changeable parameter of the graph'''

    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []
        _default_graph.variables.append(self)


class Graph():
    '''Connects nodes (placeholders, variables and operations)'''

    def __init__(self):
        self.operations = []
        self.placehoders = []
        self.variables = []

    def set_as_default(self):
        # giving access to this graph to other classes in this script
        global _default_graph
        _default_graph = self


class Session():
    '''Session of the graph, performs the operations in the correct order'''

    def run(self, operation, feed_dict={}):
        # sort nodes in the correct order of evaluation
        nodes_postorder = traverse_postorder(operation)

        for node in nodes_postorder:
            if isinstance(node, Placehoder):
                node.output = feed_dict[node]
            elif isinstance(node, Variable):
                node.output = node.value
            else:  # operation
                node.inputs = [
                    input_node.output for input_node in node.input_nodes]
                # all the operations so far expect 2 parameters but in future we can add more operations
                # that expect more than 2 parameters
                node.output = node.compute(*node.inputs)  # args parameter
            if isinstance(node.output, list):
                # convert list to numpy array
                node.output = np.array(node.output)
        return operation.output


def traverse_postorder(operation):
    '''PostOrder Traversal of Nodes. Basically makes sure computations are done in 
    the correct order.'''

    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)  # recursive call
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


class Sigmoid(Operation):
    '''Sigmoid activation function'''

    def __init__(self, z):
        super().__init__([z])

    def compute(self, z):
        return 1 / (1 + np.exp(-z))


# creates a tuple (array:array) where the first entry is the feature and the second  entry is the label
data = make_blobs(n_samples=100, n_features=2, centers=2, random_state=75)
#sfeatures = data[0]
#labels = data[1]
#plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')
# plt.show()

#x = np.linspace(0, 11, 10)
#y = -x

#np.array([1,1]).dot(np.array([[8],[10]])) - 5
#np.array([1,1]).dot(np.array([[2],[10]])) - 5

g = Graph()
g.set_as_default()
x = Placehoder() #independent variable
w = Variable([1,1]) #weights
b = Variable(-5) #bias
z = Add(Matmul(w,x),b)
a = Sigmoid(z)
sess = Session()
result = sess.run(operation=a, feed_dict={x: [2,-10]})
print(result)