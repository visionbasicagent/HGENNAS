import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')

import numpy as np
from network_designer.design_space.core.base import Model
from network_designer.design_space.extend.network import Network
from network_designer.design_space.core.search_space import calc_graph_hash

class ExtendModel(Model):
    def __init__(self, adj=None, ops=None, arch=None, design_space=None, create=True):
        super().__init__(adj, ops, arch, design_space)
        
    def create(self):
        self.model = self.create_model_from_graph(self.adj, self.ops)
        #print(self.model)
            
    def create_model_from_graph(self, adj, ops):
        #print(adj)
        o = ops.astype(int)
        a = adj.astype(int)
        a = np.triu(a, 1)
        a = a[1:,1:]
        o = o[1:]
        self.hash = calc_graph_hash(a, o)
        return Network(C=self.design_space.init_channels, 
                             N=self.design_space.N_nodes, 
                             num_classes=self.design_space.n_classes, 
                             search_space=self.design_space.candidate_operations, 
                             adj_matrix=a, ops=o)
