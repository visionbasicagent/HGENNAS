import torch 
import torch.nn as nn
#try:
from network_designer.design_space.core.net2vec import get_zc_vec
# except:
#     pass
from ptflops import get_model_complexity_info

import numpy as np

def count_parameters(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)

class Model(nn.Module):
    def __init__(self, adj=None, ops=None, arch=None, design_space=None, create=True):
        super(Model, self).__init__()
        self.adj = adj
        self.ops = ops
        self.design_space = design_space
        
        if create:
            self.create()

    def create(self):
        pass
            
    def create_model_from_graph(self, adj, ops):
        #basic funciton for our implementation create model from adjacency matrix and operation features
        raise NotImplementedError("Function 'create_model_from_graph'")
    

    def get_zc_info(self, dataloader, gpu, input_size=(3, 32, 32), delete=True, micro=True):
        # print('where is my model')
        # print(self.model)
        score = get_zc_vec(self.model, dataloader, device=gpu, micro=micro)

        # # Profile the model
        # macs, params = get_model_complexity_info(self.model, input_size, as_strings=False, print_per_layer_stat=False)
        # flops = float(macs) * 2
        
        # self.params = float(params)
        # self.flops = flops
        
        self.score = score
        #self.params = count_parameters(self.model)
        # self.flops = flops
        if delete:
            del(self.model)
            self.model = None
        return score
    
    def forward(self, inputs):
        return self.model(inputs)

    
