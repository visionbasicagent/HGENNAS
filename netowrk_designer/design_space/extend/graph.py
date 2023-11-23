from network_designer.design_space.extend.graph_base import Graph
from .operations import OPS as PRIMITIVE
import torch.nn as nn
import torch
import numpy as np
from networkx.algorithms.dag import lexicographical_topological_sort

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ResNetLikeOutput(nn.Module):
    def __init__(self, stride=1):
        super(ResNetLikeOutput, self).__init__()
        
    def forward(self, outputs):
        # sizes = []
        # for o in outputs:
        #     sizes.append(o.size())
        # print(sizes)
        return sum(outputs)

def split_integer(num, n):
    quotient = num // n
    remainder = num % n

    result = [quotient] * n
    result[-1] += remainder

    return result

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, momentum=0.1, eps=1e-5):
        super(ConvBnRelu, self).__init__()

        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_bn_relu(x)
    
class ExtendGraph(Graph):
    def __init__(self, adj_matrix, ops, search_space, channel_in, channel_out, stride=1, affine=True, track_running_stats=True, se=False): 
        super(ExtendGraph, self).__init__(adj_matrix, ops, search_space, channel_in, channel_out)

        #configuration of enter and out point
        self.se = se
        self.configure_input_output_index(0, 7)
        self.kernel_list = [1, 3, 5]
        
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.stride = stride
        self.create_nodes_from_ops(ops, adj_matrix, stride)
        

        
    def aggregate(self, x):
        return sum(x)
    
    def create_nodes_from_ops(self, ops, adj, stride):
        '''
            To do List:
                1. handle stride = 2
                2. Now its might have different C_in or C_out, so the channels and feature size can differ, we need hand first op after input
        '''
        n = adj.shape[0]  # Get the size of the square array
        indices = np.arange(n)
        zero_indices = indices[np.all(adj == 0, axis=0) & np.all(adj == 0, axis=1)]
        
        if self.stride > 1 or self.in_dim != self.out_dim:
            self.input_proj = nn.Sequential(
                                    nn.ReLU(inplace=False),
                                    nn.Conv2d(self.in_dim, self.out_dim, 1, self.stride, bias=False),
                                    nn.BatchNorm2d(self.out_dim),
                                )
        ops_kernels = ops[:, -3:]
        ops = ops[:, :-3]
        self.op_nodes = nn.ModuleDict()
        op_names = {}
        for node_idx in lexicographical_topological_sort(self.graph):
            if node_idx in zero_indices:
                op_name = 'none'
                op_names[str(node_idx)]=op_name
                node = PRIMITIVE[op_name](self.in_dim, self.out_dim, 1, stride, self.affine, self.se)
                self.op_nodes[str(node_idx)]=node
            elif node_idx == self.input_index:
                node = Identity()
                self.op_nodes[str(node_idx)]=node
            elif node_idx == self.output_index:
                self.op_nodes[str(node_idx)]=ResNetLikeOutput()
            else:
                ops_feature = ops[node_idx]
                #print(ops_feature)
                ops_kernel = ops_kernels[node_idx]
                if sum(ops_feature) == 0:
                    op_name = 'none'
                    op_names[str(node_idx)]=op_name
                    ops_kernel = self.kernel_list[np.argmax(ops_kernel)]
                else:
                    #add 1 because we have remove identity and just change in adj matrix
                    try:
                        op_name = self.search_space[np.argmax(ops_feature)+1]
                    except:
                        print(ops_feature)
                        raise TypeError
                    ops_kernel = self.kernel_list[np.argmax(ops_kernel)]
                    op_names[str(node_idx)]='_'.join([op_name, str(ops_kernel)])
                    
                if self.stride > 1 or self.in_dim != self.out_dim:
                    predecessors = [i for i in self.graph.predecessors(node_idx)]
                    if (self.input_index in predecessors) and len(predecessors) == 1:
                        #start op need to handle channels and feature_size 
                        node = PRIMITIVE[op_name](self.in_dim, self.out_dim, ops_kernel,  stride, self.affine, self.se)
                    else:
                        #later op do not handle channels and feature size 
                        node = PRIMITIVE[op_name](self.out_dim, self.out_dim,  ops_kernel,  1, self.affine, self.se)
                else:
                    #all same
                    node = PRIMITIVE[op_name](self.in_dim, self.in_dim, ops_kernel,  1, self.affine, self.se)
                self.op_nodes[str(node_idx)]=node
        #print(op_names)
        
    def forward(self, inputs):
        '''
            Ok, now we need handle differ channels, to best learn from ZenNAS, the first op with only inputs to handle differ input size and channels
        '''
        #dict to save intermediate tensor
        index_mediate_nodes = {}
        
        #enter point
        #print('new graph')
        if self.stride > 1 or self.in_dim != self.out_dim:
            index_mediate_nodes[self.input_index]=self.input_proj(inputs)
            index_mediate_nodes[-1] = inputs
        else:
            index_mediate_nodes[self.input_index]=inputs
            index_mediate_nodes[-1] = inputs
        

        for node_idx in lexicographical_topological_sort(self.graph):
            #node = self.graph.nodes[node_idx]
            predecessors = [i for i in self.graph.predecessors(node_idx)]
            if node_idx != self.output_index:
                if len(predecessors) != 0:
                    #get all predecessor output from dictionary, ignore those predecessor do not connect to input
                    #print(node_idx)
                    #print(predecessors)
                    if len(predecessors) == 1 and (predecessors[0] in index_mediate_nodes.keys()):
                        #print('are we in?')
                        if self.input_index in predecessors:
                            #print('are we in? 22 ')
                            #start node use seperate input
                            inter_x = index_mediate_nodes[-1]
                        else:
                            inter_x = index_mediate_nodes[predecessors[0]]
                    else:
                        try:
                            inter_x = self.aggregate([index_mediate_nodes[x] for x in predecessors if (x in index_mediate_nodes.keys())])
                        except:
                            #print(predecessors)
                            raise RuntimeError
                    index_mediate_nodes[node_idx] = self.op_nodes[str(node_idx)](inter_x)
            else:
                #print(predecessors)
                index_mediate_nodes[node_idx] = self.op_nodes[str(node_idx)]([index_mediate_nodes[x] for x in predecessors if (x in index_mediate_nodes.keys())])
                
            
        return index_mediate_nodes[self.output_index]