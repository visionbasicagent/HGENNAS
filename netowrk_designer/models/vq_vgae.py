
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_geometric.nn import Set2Set
from torch_geometric.data import Data, Batch

from .vgae import VGAE
from .utils import init_tensor

import torch

def gaussian_nll(y_true, mu, logvar):
    """Negative log likelihood for Gaussian distribution."""
    constant = torch.tensor(2 * torch.pi).to(y_true.device).type(y_true.dtype)
    loss = 0.5 * (torch.log(constant) + 2 * logvar) + (y_true - mu)**2 * torch.exp(-2 * logvar)
    return loss.mean()  # Calculate the mean of all loss values

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):

        flat_input = inputs
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device).double()
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize 
        quantized = torch.matmul(encodings, self._embedding.weight)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        
        return loss, quantized, perplexity, encodings

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):

        # Flatten input
        flat_input = inputs
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        ###print(f"encoding_indices: {encoding_indices.shape}")
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device).double()
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings
    
    
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True, weight_init='thomas', bias_init='thomas'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.DoubleTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.weight_init = weight_init
        self.bias_init = bias_init
        self.reset_parameters()

    def reset_parameters(self):
        init_tensor(self.weight, self.weight_init, 'relu')
        init_tensor(self.bias, self.bias_init, 'relu')

    def forward(self, adjacency, features):
        support = torch.matmul(features, self.weight)
        output = torch.bmm(adjacency, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_node, num_features):
        super(Decoder, self).__init__()

        # z -> n*n 
        self.adj_decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_node*num_node)
            ) 
        
        # z -> n*l
        self.ops_decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_node*num_features)
        )
        
        self.num_node = num_node
        self.num_features = num_features

    def forward(self, embedding):
        b, l = embedding.size()
        adj = self.adj_decoder(embedding).view(b, self.num_node, self.num_node)
        ops = self.ops_decoder(embedding).view(b, self.num_node, self.num_features)
        return adj, ops

class VQVGAE(nn.Module):
    def __init__(self, 
                num_features=1, 
                num_layers=2,
                num_hidden=32,
                num_latent=16,
                dropout_ratio=0,
                num_node=8,
                num_cond=3,
                decay=0.99, commitment_cost=0.25):

        super(VQVGAE, self).__init__()
        self.nfeat = num_features
        self.nlayer = num_layers
        self.nhid = num_hidden
        self.num_latent = num_latent
        self.dropout_ratio = dropout_ratio
        self.gc = nn.ModuleList([GraphConvolution(self.nfeat if i ==0 else self.nhid, self.nhid, bias=True) for i in range(self.nlayer)])
        self.bn = nn.ModuleList([nn.LayerNorm(self.nhid).double() for i in range(self.nlayer)])
        self.dropout = nn.ModuleList([nn.Dropout(self.dropout_ratio).double() for i in range(self.nlayer)])
        self.relu = nn.ModuleList([nn.ReLU().double() for i in range(self.nlayer)])

        self.fc = nn.Linear(self.nhid*num_node, self.num_latent).double()
        self.decoder = Decoder(self.num_latent, num_node, self.nfeat).double()
        
        self.condition_regressor = nn.Sequential(
            nn.Linear(num_latent, num_latent),
            nn.ReLU(),
            nn.Linear(num_latent, num_cond),
            nn.ReLU(),
        ).double()
        
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(64, num_latent, 
                                              commitment_cost, decay).double()
        else:
            self._vq_vae = VectorQuantizer(64, num_latent,
                                           commitment_cost).double()
            

    def forward(self, inputs):
        adjacency, features = inputs
        z = self._encoder(adjacency, features)
        vq_loss, quantized, perplexity, encodings = self._vq_vae(z)
        pred = self.condition_regressor(z)
        return z, pred, encodings
    
    def _decoder(self, emb):
        adj_recon, ops_recon = self.decoder(emb)
        return adj_recon, ops_recon
        
    def forward_decoder(self, inputs):
        adjacency, features = inputs
        z = self._encoder(adjacency, features)
        vq_loss, quantized, perplexity, encodings = self._vq_vae(z)
        pred_cond = self.condition_regressor(z)
        adj_recon, ops_recon = self.decoder(quantized)
        return adj_recon, ops_recon, z, pred_cond, vq_loss

    def _encoder(self, adjacency, features):
        
        x = self.relu[0](self.bn[0](self.gc[0](adjacency, features)))
        x = self.dropout[0](x)
        for i in range(1,self.nlayer):
            x = self.relu[i](self.bn[i](self.gc[i](adjacency, x)))
            x = self.dropout[i](x)
        
        x = x.flatten(start_dim=1)
        z = self.fc(x)

        return z


    
