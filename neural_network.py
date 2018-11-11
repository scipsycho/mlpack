import numpy as np
import pandas as pd
import logging

logging.basicConfig(filename='./log.txt',level=logging.DEBUG)
def relu(x):
    return x

class layer:

    def __init__(self, num_nodes, next_num_nodes = 1, initial_weight = None, actn_fxn = relu):

        self.num_nodes = num_nodes
        self.actn_fxn  = actn_fxn
        self.next_num_nodes = next_num_nodes

        if initial_weight is None:
            print("Using default weight values")
