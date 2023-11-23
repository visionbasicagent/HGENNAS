import torch.nn as nn
from copy import deepcopy
from .operations import ResNetBasicblock
from . graph import ExtendGraph

def get_all_layer_names_in_node(node):
    names = []
    for m in node.modules():
        names.append(m.auto_name)
    return names
    
class Network(nn.Module):
    def __init__(self, C, N, num_classes, search_space, adj_matrix, ops, args=None, affine=True, track_running_stats=True):
        super(Network, self).__init__()
        self._C = C 
        self._layerN = N
        self._num_classes = num_classes
        self._affine = affine
        self._track_running_stats = track_running_stats
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )
        
        layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ExtendGraph(adj_matrix, ops, search_space, C_prev, C_curr, stride=2)
            else:
                cell = ExtendGraph(adj_matrix, ops, search_space, C_prev, C_curr)
            self.cells.append( cell )
            C_prev = cell.out_dim
            
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(C_prev, num_classes)
        self.global_poooling = nn.AdaptiveAvgPool2d(1)
        
        self.num_node = len(list(ops))

    def forward(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)
        
        out = self.lastact(feature)
        out = self.global_poooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        
        return logits
    
    def get_node_names(self):
        
        for n, m in self.named_modules():
            m.auto_name = n
    
        node_names = {}
        cell_index = 0
        for i, cell in enumerate(self.cells):
            if isinstance(cell, ExtendGraph):
                for node_idx in range(self.num_node):
                    if node_idx not in [0, 7]:
                        if str(node_idx) in cell.op_nodes:
                            if cell_index == 0:
                                node_names[node_idx] = [get_all_layer_names_in_node(cell.op_nodes[str(node_idx)])]
                            else:
                                node_names[node_idx].append(get_all_layer_names_in_node(cell.op_nodes[str(node_idx)]))
                        else:
                            node_names[node_idx] = []
                cell_index+=1
                
        return node_names
    
    def get_block_names(self):
        
        for n, m in self.named_modules():
            m.auto_name = n
        
        block_names = []
        
        for i, cell in enumerate(self.cells):
            block_names.append(cell.auto_name)
        
        return block_names
    

    def get_flat_names(self): 
        names = []
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                names.append(n)
        return names
                    