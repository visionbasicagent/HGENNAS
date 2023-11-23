
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_geometric.nn import Set2Set
from torch_geometric.data import Data, Batch

from .utils import init_tensor

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
    
class VGAE(nn.Module):
    def __init__(self, 
                num_features=1, 
                num_layers=2,
                num_hidden=32,
                num_latent=16,
                dropout_ratio=0,
                num_node=8,
                num_cond=3):

        super(VGAE, self).__init__()
        self.nfeat = num_features
        self.nlayer = num_layers
        self.nhid = num_hidden
        self.num_latent = num_latent
        self.dropout_ratio = dropout_ratio
        self.gc = nn.ModuleList([GraphConvolution(self.nfeat if i ==0 else self.nhid, self.nhid, bias=True) for i in range(self.nlayer)])
        self.bn = nn.ModuleList([nn.LayerNorm(self.nhid).double() for i in range(self.nlayer)])
        self.dropout = nn.ModuleList([nn.Dropout(self.dropout_ratio).double() for i in range(self.nlayer)])
        self.relu = nn.ModuleList([nn.ReLU().double() for i in range(self.nlayer)])

        self.pooling = Set2Set(in_channels=num_hidden, processing_steps=4).double()
        self.fc1 = nn.Linear(self.nhid*num_node, self.num_latent).double()
        self.fc2 = nn.Linear(self.nhid*num_node, self.num_latent).double()
        self.decoder = Decoder(self.num_latent, num_node, self.nfeat).double()
        
        self.condition_regressor = nn.Sequential(
            nn.Linear(num_latent, num_latent),
            nn.ReLU(),
            nn.Linear(num_latent, num_cond),
            nn.ReLU(),
        ).double()
        
        # for m in self.modules():
        #     if isinstance(m, GraphConvolution):
        #         m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        #         # init_range = np.sqrt(6.0 / (m.input_dim + m.output_dim))
        #         # m.weight.data = torch.rand([m.input_dim, m.output_dim]).cuda()*init_range
        #         # print('find!')
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, inputs):
        adjacency, features = inputs
        mu, logvar = self._encoder(adjacency, features)
        z = self.reparameterize(mu, logvar)
        pred = self.condition_regressor(z)
        return z, pred
    
    def _decoder(self, emb):
        adj_recon, ops_recon = self.decoder(emb)
        return adj_recon, ops_recon
        
    def forward_decoder(self, inputs):
        adjacency, features = inputs
        mu, logvar = self._encoder(adjacency, features)
        z = self.reparameterize(mu, logvar)
        pred_cond = self.condition_regressor(z)
        adj_recon, ops_recon = self.decoder(z)
        return adj_recon, ops_recon, mu, logvar, z, pred_cond

    def _encoder(self, adjacency, features):
        
        x = self.relu[0](self.bn[0](self.gc[0](adjacency, features)))
        x = self.dropout[0](x)
        for i in range(1,self.nlayer):
            x = self.relu[i](self.bn[i](self.gc[i](adjacency, x)))
            x = self.dropout[i](x)
        
        x = x.flatten(start_dim=1)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        return mu, logvar
    
