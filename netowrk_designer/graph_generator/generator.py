import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')

from network_designer.models.vgae import VGAE
from network_designer.models.cnf_modules.network import GraphFlow

from network_designer.design_space.core.search_space import optimised_graph_with_lexicographical_topological_sort, expand_with_global_op
import torch
import joblib
import numpy as np
import pandas as pd 
from torch.distributions import Normal
from scipy.spatial.distance import euclidean

def post_process_adj(adj_recon):
    adj_matrix = adj_recon.detach().cpu().numpy()
    adj_matrix = np.triu(adj_matrix, 1)[0][1:,1:]
    col, row = adj_matrix.shape
    
    #print((adj_matrix >0.5).astype(int))
    for i in range(1, col):
        #from last col:
        adj_matrix[:, col-i] = (adj_matrix[:, col-i] > 0.5).astype(int)
        #print(adj_matrix)
        #if we have incoming connection from node j, j should has at-least one incoming connection
        
        #but if a intermediate node has no output we dont help them for parent
        if i==1 or np.sum((adj_matrix[col-i,:] >0.5).astype(int)) > 0:
            for j in range(1, col-i):
                # so if node has prodecessors node j, node j should has at-least one prodecessor
                if adj_matrix[j, col-i] >0 and np.sum((adj_matrix[:,j] >0.5).astype(int)) == 0:
                    select_parent = np.argmax(adj_matrix[:,j])
                    #print('add new parent {} to node {} '.format(select_parent, j))
                    adj_matrix[select_parent, j] = 1
                
                
    adj_matrix[:, 0] = (adj_matrix[:, 0] > 0.5).astype(int) 
    
    #print(adj_matrix)
    #process loss-end
    col_sums = list(np.sum(adj_matrix, axis=0))
    row_sums = list(np.sum(adj_matrix, axis=1))
    
    need_fill_output = []
    for i ,(col, row) in enumerate(zip(col_sums[:-1], row_sums[:-1])):
        #if a node has input but no output connect it to output node
        #print(i, col, row)
        if col>0 and row == 0:
            need_fill_output.append(i)
    for o in need_fill_output:
        adj_matrix[o, -1]=1
    #print(adj_matrix) 
    return adj_matrix


class GraphGenerator():
    def __init__(self, gvae_ckpt, gnf_ckpt, gmm_ckpt, gpu=torch.device("cuda:0")):
        self.gvae_ckpt = gvae_ckpt
        self.gnf_ckpt = gnf_ckpt
        self.gmm_ckpt = gmm_ckpt
        self.gpu = gpu
    
    def load_components(self):       

        self.gvae = VGAE(num_features=15, num_layers=4, num_hidden=512, num_latent=256, num_node=9, num_cond=4)

        self.gvae.load_state_dict(torch.load(self.gvae_ckpt), strict=False)
        self.gvae.eval().to(self.gpu)
        
        self.gmm = joblib.load(self.gmm_ckpt)
        
        rev_obj = torch.load(self.gnf_ckpt, map_location=self.gpu)
        self.gnf = GraphFlow(num_latent=256, num_node=9, num_cond=4)
        self.gnf.load_state_dict(rev_obj['model'].state_dict(), strict=False)
        self.gnf.to(self.gpu).double().eval()
    
    def generate_with_ref_subsetid(self, ref_id, t=0.4):
        noise = Normal(0, 1).sample(sample_shape=torch.Size([1, 256]))*t
        z = noise.to(self.gpu)
        
        with torch.no_grad():
            ref_cond = self.gmm.means_[ref_id]
            ref_cond = torch.tensor(ref_cond).double().to(self.gpu)
            #print(ref_cond)

            samples = self.gnf.sample(z.to(self.gpu).double(),ref_cond)
            adj, features = self.gvae._decoder(samples)
            
        adj=torch.nn.Sigmoid()(adj)
        features = torch.nn.Sigmoid()(features)
 
        # #print(features)
        adj_matrix = post_process_adj(adj)
        adj_matrix = optimised_graph_with_lexicographical_topological_sort(adj_matrix)
        adj_matrix_size = np.shape(adj_matrix)[1]
        adj_matrix = expand_with_global_op(adj_matrix_size, adj_matrix)
        #adj_matrix = adj.cpu().numpy()

        ops_f = features[:, :, :-3]
        #print(ops_f.size())
        ops_k = features[:, :, -3:]
        ops_f_mask = (torch.sum(ops_f, dim=-1) > 0).int()[:, :, None]
        ops_k_mask = (torch.sum(ops_k, dim=-1) > 0).int()[:, :, None]
        ops_feature = torch.mul(torch.nn.functional.one_hot(ops_f.argmax(dim=-1) , num_classes=12), ops_f_mask)
        ops_kernel = torch.mul(torch.nn.functional.one_hot(ops_k.argmax(dim=-1) , num_classes=3), ops_k_mask)
        ops_feature = torch.cat([ops_feature, ops_kernel], dim=-1)
          
        return adj_matrix, ops_feature.cpu().numpy()[0]
            
        
        
    def generate_with_ref_graph(self, ref_adj=None, ref_ops=None, t=0.2):
        
        #print('in gen')
        noise = Normal(0, 1).sample(sample_shape=torch.Size([1, 256]))*t
        z = noise.to(self.gpu)
        
        if (ref_adj is not None) and (ref_ops is not None):
            #print('in con')
            ref_adj = torch.tensor(ref_adj).to(self.gpu).double().unsqueeze(0)
            ref_ops = torch.tensor(ref_ops).to(self.gpu).double().unsqueeze(0)
            with torch.no_grad():
                embs, pred_cond = self.gvae((ref_adj, ref_ops))
                ref_label = self.gmm.predict(pred_cond.cpu().numpy())
                print(ref_label)

                #ref_cond = self.gmm.means_[ref_label]
                #ref_cond = torch.tensor(ref_cond).double().cuda()
                #print(ref_cond)
                #ref_cond = torch.tensor()
                samples = self.gnf.sample(z.to(self.gpu).double(),pred_cond)
                #print(self.gmm.predict(samples.flatten(start_dim=1).cpu().numpy()))
                adj, features = self.gvae._decoder(samples)
 
                adj=torch.nn.Sigmoid()(adj)
                features = torch.nn.Sigmoid()(features)
        
                # #print(features)
                adj_matrix = post_process_adj(adj)
                adj_matrix = optimised_graph_with_lexicographical_topological_sort(adj_matrix)
                adj_matrix_size = np.shape(adj_matrix)[1]
                adj_matrix = expand_with_global_op(adj_matrix_size, adj_matrix)
                #adj_matrix = adj.cpu().numpy()

                ops_f = features[:, :, :-3]
                #print(ops_f.size())
                ops_k = features[:, :, -3:]
                ops_f_mask = (torch.sum(ops_f, dim=-1) > 0).int()[:, :, None]
                ops_k_mask = (torch.sum(ops_k, dim=-1) > 0).int()[:, :, None]
                ops_feature = torch.mul(torch.nn.functional.one_hot(ops_f.argmax(dim=-1) , num_classes=12), ops_f_mask)
                ops_kernel = torch.mul(torch.nn.functional.one_hot(ops_k.argmax(dim=-1) , num_classes=3), ops_k_mask)
                ops_feature = torch.cat([ops_feature, ops_kernel], dim=-1)
                
                return adj_matrix, ops_feature.cpu().numpy()[0]
    
    def generate_with_VAE_free(self):
        
        noise = Normal(0, 1).sample(sample_shape=torch.Size([1, 256]))
        z = noise.cuda()
        
        with torch.no_grad():
            adj, features = self.gvae._decoder(z.cuda().double())

            adj=torch.nn.Sigmoid()(adj)
            features = torch.nn.Sigmoid()(features)
    
            # #print(features)
            adj_matrix = post_process_adj(adj)
            adj_matrix = optimised_graph_with_lexicographical_topological_sort(adj_matrix)
            adj_matrix_size = np.shape(adj_matrix)[1]
            adj_matrix = expand_with_global_op(adj_matrix_size, adj_matrix)
            #adj_matrix = adj.cpu().numpy()

            ops_f = features[:, :, :-3]
            #print(ops_f.size())
            ops_k = features[:, :, -3:]
            ops_f_mask = (torch.sum(ops_f, dim=-1) > 0).int()[:, :, None]
            ops_k_mask = (torch.sum(ops_k, dim=-1) > 0).int()[:, :, None]
            ops_feature = torch.mul(torch.nn.functional.one_hot(ops_f.argmax(dim=-1) , num_classes=12), ops_f_mask)
            ops_kernel = torch.mul(torch.nn.functional.one_hot(ops_k.argmax(dim=-1) , num_classes=3), ops_k_mask)
            ops_feature = torch.cat([ops_feature, ops_kernel], dim=-1)
            
            return adj_matrix, ops_feature.cpu().numpy()[0]
    
    def generate_with_VAE_ref_graph(self, ref_adj=None, ref_ops=None, t=0.2):
        
        #print('in gen')
        noise = Normal(0, 1).sample(sample_shape=torch.Size([1, 256]))*t
        z = noise.cuda()
        
        if (ref_adj is not None) and (ref_ops is not None):
            #print('in con')
            ref_adj = torch.tensor(ref_adj).cuda().double().unsqueeze(0)
            ref_ops = torch.tensor(ref_ops).cuda().double().unsqueeze(0)
            with torch.no_grad():
                embs, pred_cond = self.gvae((ref_adj, ref_ops))
                
                embs = embs + z
                
                samples = self.gnf.sample(z.cuda().double(),pred_cond)
                #print(self.gmm.predict(samples.flatten(start_dim=1).cpu().numpy()))
                adj, features = self.gvae._decoder(samples)
 
                adj=torch.nn.Sigmoid()(adj)
                features = torch.nn.Sigmoid()(features)
        
                # #print(features)
                adj_matrix = post_process_adj(adj)
                adj_matrix = optimised_graph_with_lexicographical_topological_sort(adj_matrix)
                adj_matrix_size = np.shape(adj_matrix)[1]
                adj_matrix = expand_with_global_op(adj_matrix_size, adj_matrix)
                #adj_matrix = adj.cpu().numpy()

                ops_f = features[:, :, :-3]
                #print(ops_f.size())
                ops_k = features[:, :, -3:]
                ops_f_mask = (torch.sum(ops_f, dim=-1) > 0).int()[:, :, None]
                ops_k_mask = (torch.sum(ops_k, dim=-1) > 0).int()[:, :, None]
                ops_feature = torch.mul(torch.nn.functional.one_hot(ops_f.argmax(dim=-1) , num_classes=12), ops_f_mask)
                ops_kernel = torch.mul(torch.nn.functional.one_hot(ops_k.argmax(dim=-1) , num_classes=3), ops_k_mask)
                ops_feature = torch.cat([ops_feature, ops_kernel], dim=-1)
                
                return adj_matrix, ops_feature.cpu().numpy()[0]
    