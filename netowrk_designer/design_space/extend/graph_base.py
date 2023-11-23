import networkx as nx
import torch.nn as nn
import numpy as np

from networkx.algorithms.dag import lexicographical_topological_sort
    
class Graph(nn.Module):
    def __init__(self, adj_matrix, ops, search_space, channel_in, channel_out):
        super(Graph, self).__init__()
        adj_matrix = np.triu(adj_matrix, 1)
        self.graph = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)
        self.in_dim = channel_in
        self.out_dim = channel_out
        
        #configuration of enter and out point
        self.input_index = 0
        self.output_index = 7
        
        self.search_space = search_space
        
        # #debug
        # for node_idx in lexicographical_topological_sort(self.graph):
        #     predecessors = [i for i in self.graph.predecessors(node_idx)]
        #     if len(predecessors) != 0:
        #         print(node_idx, predecessors)
        
    def create_nodes_from_ops(self, ops):
        pass
    
    def aggregate(self, x):
        pass
    
    def configure_input_output_index(self, input, output):
        self.input_index = input
        self.output_index = output
    
    def forward(self, inputs):
        pass
        