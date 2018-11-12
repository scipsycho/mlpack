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

    def __init__(self, num_nodes, next_num_nodes = 1, initial_weight = None, is_last = False, has_bias = False, actn_fxn = relu):

        self.num_nodes = num_nodes
        self.actn_fxn  = actn_fxn
        self.next_num_nodes = next_num_nodes
        self.is_last = is_last
        self.weight_matrix = None
        self.has_bias = has_bias

        #cannot have a bias in output layer
        if self.has_bias and self.is_last:
            logging.critical("An output layer cannot have a bias. Cannot Continue")
            exit()
        
        #cannot have a bias only in a layer (it doesn't make sense)
        if self.has_bias and self.num_nodes == 1:
            logging.critical("A layer cannot have only one node that is a bias")
            exit()
        
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

        suppossed_input_vector_size = self.num_nodes
        
        if self.has_bias:
            suppossed_input_vector_size -= 1
            
        if input_vector.shape != (suppossed_input_vector_size,):
            logging.critical("Input vector not of desired size. Cannot continue!")
            return None

        temp = []
        if self.has_bias:
            temp.append(1)
        
        for i in input_vector:
            temp.append(self.actn_fxn(i))
        self.input_vector = np.asarray([ self.actn_fxn(i) for i in input_vector ])

        return self.input_vector

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

    def __init__(self, ff_layers, initial_weights = None, has_bias = None,actn_fxn = relu):

        self.ff_layers = list(np.array(ff_layers).flatten())
        self.num_layers = len(self.ff_layers)

        if self.num_layers < 2:
            logging.critical('Neural Network cannot have less than two layers. Cannot Continue')
            exit()
        
        self.n_layers = []
        
        if has_bias is None:
            has_bias = [False for i in range(self.num_layers-1)]
        elif np.asarray(has_bias).flatten().shape != (self.num_layers-1,):
            logging.warning('has_bias is not of compatible size. Using default value')
            has_bias = [False for i in range(self.num_layers-1)]
        
        if isinstance(initial_weights,(type(None),int,float)):
            self.n_layers = [ layer(num_nodes = self.ff_layers[i], next_num_nodes= self.ff_layers[i+1],
                                    actn_fxn = actn_fxn, initial_weight= initial_weights, has_bias = has_bias[i])
                               for i in range(self.num_layers - 1)]

        elif initial_weights.shape[0] == self.num_layers - 1:
            self.n_layers = [ layer(num_nodes = self.ff_layers[i], next_num_nodes= self.ff_layers[i+1],
                                    actn_fxn = actn_fxn, initial_weight= initial_weights[i], has_bias = has_bias[i])
                               for i in range(self.num_layers - 1)]

        self.n_layers.append(layer(num_nodes = self.ff_layers[self.num_layers-1],initial_weight=1,
                                       next_num_nodes=self.ff_layers[self.num_layers-1], actn_fxn = actn_fxn, is_last = True,
                                  has_bias = False))

        self.n_layers[0].actn_fxn = identity
        self.input_vector = None
        self.output_vector = None
        
        #default learning algos
        self.learning_algos = {}
        self.learning_algos['hebb'] = self.hebb_learn
        #self.learning_algos['perceptron'] = self.perceptron_learn
        #self.learning_algos['delta'] = self.delta_learn
        #self.learning_algos['backprop'] = self.backprop

    def send(self, input_vector):
        
        #checking if the input layer of neural network accepts the input_vector as a valid input vector
        return_val = self.n_layers[0].send(input_vector)
        
        if return_val is None:
            logging.critical('Input_vector incompatible. Cannot continue')
            return None;
        
        #flattened to keep things simple
        self.input_vector = np.array(input_vector).flatten()
        
        return self.input_vector
    
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

        
        #legend
        dot.attr(rankdir='LR',ranksep='4')
        dot.node(name="input_layer", rank = 'sink',xlabel="Input Layer",label="", fontsize = '12',style="filled",color="green",fixedsize = 'true', shape="square",width="0.1")
        dot.node(name="output_layer", xlabel="Output Layer",label="", fontsize = '12',style="filled",color="red",fixedsize = 'true', shape="square",width="0.1")
        dot.node(name="hidden_layers", xlabel="Hidden Layers",label="", fontsize = '12',style="filled",color="grey",fixedsize = 'true', shape="square",width="0.1")
        
        start = 0
        for i in range(self.num_layers):
            color = "grey"
            if i == 0:
                color = "green"
            elif i == self.num_layers - 1:
                color = "red"
                
            with dot.subgraph(name = 'cluster_'+str(i)) as subdot:
                for j in range(self.n_layers[i].num_nodes):
                    label_name = 'x' + str(i) + str(j)
                    if self.n_layers[i].has_bias and j == 0:
                        label_name = '1'
                
                    subdot.node(name = str(start),label = label_name,color=color, style="filled",rankdir='TB')
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

        #dot.render('./'+comment+'.png', view = True)
        dot.view()
        
    def hebb_learn(self,arg_dict):
        output_vector = arg_dict['output_vector']
        
        #hebb only works when there are no hidden layers
        if self.num_layers > 2:
            logging.critical('Hebb cannot work with any hidden layers. Cannot Continue')
            return None
        
        input_layer = self.n_layers[0]
        for i in range(len(output_vector)):
            output = output_vector[i]
            for j in range(input_layer.num_nodes):
                input_layer.weight_matrix[j][i] += output*self.input_vector[j]
        
        
    def learn(self, learning_algo: str, input_matrix, output_matrix, epochs = 1, **xargs):
        
        if learning_algo not in self.learning_algos.keys():
            logging.critical('{} learning algorithm is not recoginized. Cannot Continue')
            return None
        
        input_matrix = np.array(input_matrix)
        output_matrix = np.array(output_matrix)
        
        if len(input_matrix.shape) != 2 or len(output_matrix.shape) != 2:
            logging.critical('Input and Output matrices should have only two dimensions. Cannot Continue')
            return None
        
        if input_matrix.shape[0] != output_matrix.shape[0]:
            logging.critical('Input and Output matrices have incompatible dimensions. Cannot Continue')
            return None
        
        input_layer = self.n_layers[0]
        output_layer = self.n_layers[self.num_layers-1]
        
        num_pairs = input_matrix.shape[0]
        
        if num_pairs < 1:
            logging.critical('No input provided. Cannot Continue')
            return None
        
        if input_layer.send(input_matrix[0]) is None:
            logging.critical('Input Matrix dimensions` incompatible. Cannot Continue')
            return None
        
        if output_layer.send(output_matrix[0]) is None:
            logging.critical('Output Matrix dimensions` incompatible. Cannot Continue')
            
        
        for i in range(epochs):
            for j in range(num_pairs):
                self.send(input_matrix[j].flatten())
                self.generate()
                xargs['output_vector'] = output_matrix[j].flatten()
                self.learning_algos[learning_algo](xargs)
        
        
