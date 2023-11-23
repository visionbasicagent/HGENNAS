import torch.nn as nn
from copy import deepcopy
from .operations import ResNetBasicblock
from .graph import ExtendGraph
import numpy as np
import torch.nn.functional as F
import torch
import logging
def get_all_layer_names_in_node(node):
    names = []
    for m in node.modules():
        names.append(m.auto_name)
    return names
    
class MasterNetwork(nn.Module):
    def __init__(self, 
                 layer_channels, 
                 layer_stride, 
                 layer_repeats, 
                 final_concat_layer, 
                 num_classes, 
                 search_space,
                 adj_matrixs, 
                 opss,
                 args=None,
                 affine=True, 
                 track_running_stats=True,
                 shave=False, 
                 se=False,
                 stem_stride=1,
                 dataset='cifar10'):

        super(MasterNetwork, self).__init__()
        self._num_classes = num_classes
        self._affine = affine
        self._track_running_stats = track_running_stats
        self.stem = nn.Sequential(
            nn.Conv2d(3, layer_channels[0], kernel_size=3, padding=1, bias=False, stride=stem_stride),
            nn.BatchNorm2d(layer_channels[0])
        )
        

        C_prev = layer_channels[0]
        self.cells = nn.ModuleList()
        for index, (C_curr, stride) in enumerate(zip(layer_channels, layer_stride)):
            current_stride = stride
            for _ in range(layer_repeats[index]):
                adj = adj_matrixs[index]
                ops = opss[index]
                if shave:
                    adj = np.triu(adj, 1)
                    adj = adj[1:,1:]
                    ops = ops[1:]

                cell = ExtendGraph(adj, ops, search_space, C_prev, C_curr, stride=current_stride, se=se)
                self.cells.append( cell )
                C_prev = cell.out_dim
                current_stride = 1
        #logging.info(num_classes)
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), 
            nn.ReLU(inplace=True),

            nn.Conv2d(C_prev, final_concat_layer, kernel_size=1, bias=False),
            nn.BatchNorm2d(final_concat_layer))
        self.classifier = nn.Linear(final_concat_layer, num_classes)
        self.global_poooling = nn.AdaptiveAvgPool2d(1)
        
        self.num_node = len(list(opss[0]))
        
        #self.init_parameters()

    # def init_parameters(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.xavier_normal_(m.weight.data, gain=3.26033)
    #             if hasattr(m, 'bias') and m.bias is not None:
    #                 nn.init.zeros_(m.bias)
    #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #             if (m.weight is not None) and (m.bias is not None):
    #                 nn.init.ones_(m.weight)
    #                 nn.init.zeros_(m.bias)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 3.26033 * np.sqrt(2 / (m.weight.shape[0] + m.weight.shape[1])))
    #             if hasattr(m, 'bias') and m.bias is not None:
    #                 nn.init.zeros_(m.bias)
    #         else:
    #             pass

        #self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=3.26033)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if (m.weight is not None) and (m.bias is not None):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 3.26033 * np.sqrt(2 / (m.weight.shape[0] + m.weight.shape[1])))
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                pass


    def forward(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)
        
        out = self.lastact(feature)
        out = self.global_poooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        
        return logits

    def extract_stage_features_and_logit(self, x, target_downsample_ratio=None):
        stage_features_list = []
        image_size = x.shape[2]
        output = self.stem(x)
        #stage_features_list.append(output)
        

        for block_id, the_block in enumerate(self.cells):
            output = the_block(output)
            dowsample_ratio = round(image_size / output.shape[2])
            if dowsample_ratio == target_downsample_ratio:
                stage_features_list.append(output)
                target_downsample_ratio *= 2
            pass
        pass

        out = self.lastact(output)
        out = self.global_poooling(out)
        out = out.view(out.size(0), -1)
        logit = self.classifier(out)
        return stage_features_list, logit

import sys
import random
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from network_designer.design_space.extend.search_space import SearchSpace

def generate_random_arch(layer_budget=20):
    
    channel_base = 8
    channel_range = 32
    
    stride_option = [1, 2]
    
    repeats_range = 4
    #graph encoding
    ss = SearchSpace(6, 10)
    layer_channels = []
    layer_stride = []
    layer_repeats = []
    adj_matrixs = []
    opss = []
    layer_budget = random.randint(5, layer_budget)
    while layer_budget > 0:
        adj, ops = ss.sample()
        ch  = random.randint(1, channel_range) * channel_base
        std = random.choice(stride_option)
        
        repeats = random.randint(1, min(repeats_range, layer_budget))
        
        adj_matrixs.append(adj)
        opss.append(ops)
        layer_channels.append(ch)
        layer_repeats.append(repeats)
        layer_stride.append(std)
       
        layer_budget -= repeats
        
    return layer_channels, layer_stride, layer_repeats, adj_matrixs, opss