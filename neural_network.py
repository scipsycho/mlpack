import numpy as np
import pandas as pd
import logging
from graphviz import Digraph
import math

logging.basicConfig(filename='./log.txt',level=logging.DEBUG)


def sigmoid_bipolar(x):
    return sigmoid_binary(x) - 0.5
def sigmoid_binary(x):
    return  1/(1 + math.e**(-x))
def relu(x):
    logging.debug("Relu({}) Called: returning {}".format(x,max(x,0)))
    return max(0,x)

def identity(x):
    logging.debug("Identity({}) Called: returning {}".format(x,x))
    return x

def step(x,theta):
    ret = 0
    if x > theta:
        ret = 1
    elif x < theta:
        ret = -1
    else:
        ret = 0

    logging.debug("Step({},{}) Called: returning {}".format(x,theta,ret))

class layer:

    def __init__(self, num_nodes, next_num_nodes = 1, initial_weight = None, is_last = False, actn_fxn = relu, has_bias = False):

        self.num_nodes = num_nodes
        self.actn_fxn  = actn_fxn
        self.next_num_nodes = next_num_nodes
        self.is_last = is_last
        self.weight_matrix = None

        logging.debug("layer object initialized with \n\tNumber of nodes: {}\n\tActivation Function: {}\n\tNext Number of Nodes: {}\n\tIs Last: {}\n".format(self.num_nodes,self.actn_fxn,self.next_num_nodes,self.is_last))
        if initial_weight is None:
            logging.debug('Using default values for the weight matrix')
            self.weight_matrix = np.zeros([self.num_nodes, self.next_num_nodes],dtype=float)

        elif isinstance(initial_weight, (int,float)):
            logging.debug("Using the weight value for all weights in the layer")
            self.weight_matrix = (np.zeros(shape = (self.num_nodes, self.next_num_nodes),dtype=float))
            self.weight_matrix.fill(float(initial_weight))

        elif initial_weight.shape == (self.num_nodes, self.next_num_nodes):
            logging.debug("Using user provided values")
            self.weight_matrix = np.array(initial_weight, dtype = float)



        else:
            logging.warning('Weight matrix provided is of incompatible size. Using default values')
            self.weight_matrix = np.zeros(shape = (self.num_nodes,self.next_num_nodes),dtype = float)

        self.input_vector = None

    def send(self, input_vector):

        #flattened to keep things simple
        input_vector = np.array(input_vector).flatten()

        if input_vector.shape != (self.num_nodes,):
            logging.critical("Input vector not of desired size. Cannot continue!")
            return None

        self.input_vector = np.asarray([ self.actn_fxn(i) for i in input_vector ])


    def generate(self):

        if self.input_vector is None:
            logging.critical("Input not yet provided. Cannot continue!")
            return None

        if self.is_last:
            return self.input_vector

        #the input is required to be in a row only
        temp_input_matrix = np.reshape(self.input_vector, newshape=(1,self.num_nodes))

        #input is consumed
        self.input_vector = None

        output_matrix = np.dot(temp_input_matrix,self.weight_matrix)

        return output_matrix.flatten()


    def desc(self):
        print("Number of Nodes: {}\n Number of Nodes in next layer: {}\nWeights:\n {}\n"
               .format(self.num_nodes, self.next_num_nodes,self.weight_matrix))



class n_ff_network:

    def __init__(self, ff_layers, initial_weights = None, actn_fxn = relu):

        self.ff_layers = list(np.array(ff_layers).flatten())
        self.num_layers = len(self.ff_layers)

        self.n_layers = []
        if isinstance(initial_weights,(type(None),int,float)):
            self.n_layers = [ layer(num_nodes = self.ff_layers[i], next_num_nodes= self.ff_layers[i+1],
                                    actn_fxn = actn_fxn, initial_weight= initial_weights)
                               for i in range(self.num_layers - 1)]

        elif initial_weights.shape[0] == self.num_layers - 1:
            self.n_layers = [ layer(num_nodes = self.ff_layers[i], next_num_nodes= self.ff_layers[i+1],
                                    actn_fxn = actn_fxn, initial_weight= initial_weights[i])
                               for i in range(self.num_layers - 1)]

        self.n_layers.append(layer(num_nodes = self.ff_layers[self.num_layers-1],initial_weight=1,
                                       next_num_nodes=self.ff_layers[self.num_layers-1], actn_fxn = actn_fxn, is_last = True))

        self.n_layers[0].actn_fxn = identity
        self.input_vector = None
        self.output_vector = None

    def send(self, input_vector):
        #flattened to keep things simple
        input_vector = np.array(input_vector).flatten()

        if input_vector.shape != (self.n_layers[0].num_nodes,):
            logging.critical("Input vector not of desired size. Cannot continue!")
            return None

        self.input_vector = input_vector

    def generate(self):

        curr_input = self.input_vector

        count = 0
        for i in self.n_layers:
            count +=1
            i.send(curr_input)
            curr_input = i.generate()

        self.output_vector = curr_input

        return self.output_vector

    def desc(self):
        for i in range(self.num_layers):
            print("Layer {}\n".format(i+1))
            self.n_layers[i].desc()

    def show_network(self,comment="Neural Network"):
        dot = Digraph(comment)

        start = 0

        for i in range(self.num_layers):
            for j in range(self.n_layers[i].num_nodes):
                dot.node(name = str(start),label = 'x'+str(i)+str(j))
                start +=1

        start = 0
        for i in range(self.num_layers-1):
            curr_layer = self.n_layers[i]
            next_start = start + curr_layer.num_nodes

            tail = start
            head = next_start
            for j in range(curr_layer.num_nodes):
                head = next_start
                for k in range(curr_layer.next_num_nodes):
                    dot.edge(tail_name = str(tail),head_name = str(head), label = str(curr_layer.weight_matrix[j][k]))
                    head += 1
                tail += 1

            start = next_start

        dot.render('./'+comment+'.png', view = True)
