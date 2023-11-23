import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
import copy
import random
import numpy as np
from network_designer.design_space.core.base import Model
from network_designer.design_space.extend.masternetwork import MasterNetwork
from network_designer.design_space.extend.operations import SearchSpaceNames
from tqdm import tqdm
from network_designer.design_space.extend.search_space import SearchSpace
from network_designer.design_space.core.search_space import mutate_arch
search_space = SearchSpaceNames['nf_graph']
ss = SearchSpace(6, 10)

def random_subset(input_list):
    subset_length = random.randint(1, len(input_list))
    return random.sample(input_list, subset_length)


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class MasterModel(Model):
    def __init__(self,  layer_channels, layer_stride, layer_repeats, 
                 final_concat_layer, adjs=None, opss=None, arch=None, design_space=None, 
                 create=True, shave=True, num_classes=10, se=False, stem_stride=1,
                 dataset='cifar10'):
        super().__init__(adjs, opss, arch, design_space,create=False)
        
        self.layer_channels = layer_channels
        self.layer_stride = layer_stride
        self.layer_repeats = layer_repeats
        
        self.adjs = adjs 
        self.opss = opss 
        self.n_classes = num_classes
        self.shave = shave
        self.se = se
        self.num_classes=num_classes
        self.stem_stride = stem_stride

        self.final_concat_layer = final_concat_layer
        self.dataset = dataset
        if create:
            self.create()

            
            
    def create(self):
        self.model = self.create_model_from_graph(self.layer_channels, 
                                                  self.layer_stride, 
                                                  self.layer_repeats, 
                                                  self.final_concat_layer, 
                                                  self.adjs, 
                                                  self.opss,
                                                  self.dataset)
        #print(self.model)
            
    def create_model_from_graph(self, 
                                layer_channels, 
                                layer_stride, 
                                layer_repeats, 
                                final_concat_layer, 
                                adj, 
                                ops,
                                dataset='cifar10'):
        return MasterNetwork(layer_channels, layer_stride, layer_repeats, final_concat_layer, num_classes=self.n_classes, search_space=search_space, adj_matrixs=adj, opss=ops, shave=self.shave, se=self.se, stem_stride=self.stem_stride)

    def mutate_graph(self, adj, ops):
        return mutate_arch(adj, ops)
    
    def gmm_guided_mutate(self, adj, ops, gmm, gvae):
        pass
    
   #def 
    def mutate(self, generator=None, cell_mutate_strategy='RESAMPLE'):
        
        channel_base = 8
        channel_range = 128
        stride_option = [1, 2]
        repeats_range = 8
        mutation_types = ['cellstructure_mutation', 'channels_mutation', 'stride_mutation', 'repeats_mutation', 'last_concat_mutation']
        mutation_type = random.choice(mutation_types)
        print(mutation_type, cell_mutate_strategy)
        #print(cell_mutate_strategy == 'GMM_GUIDE')
        mutation_idx = random.choice(list(range(len(self.adjs))))
        
        layer_channels = copy.deepcopy(self.layer_channels)
        layer_stride = copy.deepcopy(self.layer_stride)
        layer_repeats = copy.deepcopy(self.layer_repeats)
        
        adjs = copy.deepcopy(self.adjs)
        opss = copy.deepcopy(self.opss)
        
        final_concat_layer = self.final_concat_layer
        mutation_actions = random_subset(mutation_types)
        
        if 'cellstructure_mutation' in mutation_actions:
            #print('cell')
            if cell_mutate_strategy == 'RESAMPLE':
                adj, ops = ss.sample()
            elif cell_mutate_strategy == 'CELLMUTATE':
                adj, ops = self.mutate_graph(adjs[mutation_idx], opss[mutation_idx])
            elif cell_mutate_strategy == 'GMM_GUIDE':
                #print('gen gmm')
                adj, ops = generator.generate_with_ref_graph(ref_adj=adjs[mutation_idx], ref_ops=opss[mutation_idx], t=1.0)
                
            adjs[mutation_idx] = adj
            opss[mutation_idx] = ops
        
        if 'channels_mutation' in mutation_actions:
            multi = random.choice([2.5, 2, 1.5, 1.25, 1, 1/1.25, 1/1.5, 1/2, 1/2.5])
            target = layer_channels[mutation_idx] * multi
            layer_channels[mutation_idx] = clamp(target, channel_base, channel_base*channel_range)
            layer_channels[mutation_idx] = int(layer_channels[mutation_idx] - layer_channels[mutation_idx] % 8)
            print('mutated channel {} after mutate multi {}'.format(layer_channels[mutation_idx], multi))
        
        if 'stride_mutation' in mutation_actions:
            layer_stride[mutation_idx] = random.choice(stride_option)
        
        if 'repeats_mutation' in mutation_actions:
            mutate = random.choice([0, 1, 2, -1, -2])
            
            remain_budgets = 20 - (sum(layer_repeats) + mutate)
            if remain_budgets > 0:
                new_repeats=  max(0, min(layer_repeats[mutation_idx]+mutate, remain_budgets))
                
                if new_repeats == 0:
                    if len(layer_repeats) > 1:
                        layer_channels.pop(mutation_idx) 
                        layer_repeats.pop(mutation_idx) 
                        layer_stride.pop(mutation_idx) 
                        adjs.pop(mutation_idx)
                        opss.pop(mutation_idx)
                elif new_repeats >4 :
                    layer_repeats.insert(mutation_idx+1, new_repeats-4)
                    layer_channels.insert(mutation_idx+1, layer_channels[mutation_idx]) 
                    layer_stride.insert(mutation_idx+1, layer_stride[mutation_idx]) 
                    adjs.insert(mutation_idx+1, adjs[mutation_idx]) 
                    opss.insert(mutation_idx+1, opss[mutation_idx]) 
                else:
                    layer_repeats[mutation_idx] = new_repeats
        
        if 'last_concat_mutation' in mutation_actions:
            multi = random.choice([2.5, 2, 1.5, 1.25, 1, 1/1.25, 1/1.5, 1/2, 1/2.5])
            start = final_concat_layer
            target = final_concat_layer * multi
            final_concat_layer = clamp(target, channel_base, channel_base*channel_range)
            final_concat_layer = int(final_concat_layer - final_concat_layer % 8)
            print('mutated channel from{} to  {} after mutate multi {}'.format(start, final_concat_layer, multi))
        
        
                
            
        return layer_channels, layer_stride, layer_repeats, adjs, opss, final_concat_layer