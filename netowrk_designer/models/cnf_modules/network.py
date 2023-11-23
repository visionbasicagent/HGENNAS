import os
import sys
import torch
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from torch import nn
#from network_designer.model.zerocostnet import ZeroCostNetContinues
from network_designer.models.cnf_modules.flow import cnf
import torch.nn.functional as F

class GraphFlow(nn.Module):
    def __init__(self, num_latent, num_node, dims="512-512-512-512-512-512-512-512-512", num_cond=2):
        super(GraphFlow, self).__init__()
        print(num_cond)
        self.graph_cnf = cnf(num_latent, dims,num_cond, 1)
        
        
    def forward(self, embs, ref_zc, logx):
        approx21, delta_log_p2 = self.graph_cnf(embs, ref_zc, logx)

        return approx21, delta_log_p2
    
    def sample(self, z, ref_zc):
        z = self.graph_cnf(z, ref_zc, reverse=True)
        return z