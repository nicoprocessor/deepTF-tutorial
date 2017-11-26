import numpy as np


class Operation():

    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)

    def compute(self, x, y):
        # to be overwritten by extending classes
        pass


class Add(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        # override compute method from Operation class
        self.inputs = [x, y]
        return x + y  # perform sum


class Multiply(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        # override compute method from Operation class
        self.inputs = [x, y]
        return x * y  # perform scalar multiplication


class Matmul(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        # override compute method from Operation class
        self.inputs = [x, y]
        return x.dot(y)  # perform matrix multiplication
