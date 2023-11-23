import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
import copy
import random
import math
import numpy as np
from tqdm import tqdm
import logging
from network_designer.design_space.extend.mastermodel import MasterModel
from network_designer.design_space.extend.operations import SearchSpaceNames
from network_designer.design_space.extend.search_space import SearchSpace
from network_designer.design_space.core.search_space import mutate_arch


from ptflops import get_model_complexity_info
search_space = SearchSpaceNames['nf_graph']
ss = SearchSpace(6, 10)

def random_subset(input_list):
    subset_length = random.randint(1, len(input_list))
    return random.sample(input_list, subset_length)


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class MasterSpace:

    def __init__(self,  layer_channels, layer_stride, layer_repeats, 
                 final_concat_layer, graphs, evaluate_size=50, shave=True, n_classes=10, 
                 stem_stride=1,dataset='cifar10'):
     
        self.layer_channels = layer_channels
        self.layer_stride = layer_stride
        self.layer_repeats = layer_repeats
        self.final_concat_layer = final_concat_layer
        
        self.graphs = graphs  
        self.n_classes = n_classes
        self.shave = shave
        self.evaluate_size = evaluate_size
        #logging.info(self.n_classes)
        self.population = []
        self.stem_stride = stem_stride
        self.dataset = dataset
    
    def generate_graph(self, generator, space_idx):
        return generator.generate_with_ref_subsetid(ref_id=space_idx, t=0.4)
    
    def get_average_population_statistics(self):
        avg_flops = 0
        avg_param = 0
        avg_zc_1 = 0
        #avg_zc_2 = 0
        for p in self.population:
            avg_flops += p.flops
            avg_param += p.params
            avg_zc_1 += p.score[0]
            #avg_zc_2 += p.score[1]
        
        self.avg_flops = avg_flops/len(self.population)
        self.avg_param = avg_param/len(self.population)
        self.avg_zc_1 = avg_zc_1/len(self.population)
        #self.avg_zc_2 = avg_zc_2/len(self.population)
            
        
    def evaluate_space(self, generator, input_size, gpu, dataloader):
        successful_sample = 0
        
        while successful_sample < self.evaluate_size:
            gen_adjs = []
            gen_opss = []
            for g_idx in self.graphs:
                a, o = self.generate_graph(generator, int(g_idx))
                gen_adjs.append(a)
                gen_opss.append(o)
            

            model = MasterModel(self.layer_channels, 
                                self.layer_stride, 
                                self.layer_repeats, 
                                self.final_concat_layer, 
                                adjs=gen_adjs, 
                                opss=gen_opss, 
                                num_classes=self.n_classes, 
                                stem_stride=self.stem_stride,
                                dataset=self.dataset).to(device=gpu)

            
            try:
                macs, params = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
                flops = float(macs) * 2
                params = float(params)
                model.get_zc_info(dataloader, gpu=gpu, micro=False, input_size=input_size)
                if math.isnan(model.score[0]):
                    continue
                model.flops = flops
                model.params = params
                self.population.append(model)
                successful_sample += 1
            except:
                continue
                 
    def mutate(self):
        
        channel_base = 8
        channel_range = 128
        stride_option = [1, 2]
        repeats_range = 8
        mutation_types = ['cell_mutation', 'channels_mutation', 'stride_mutation', 'repeats_mutation', 'last_concat_mutation']
        
        mutation_idx = random.choice(list(range(len(self.graphs))))
        
        layer_channels = copy.deepcopy(self.layer_channels)
        layer_stride = copy.deepcopy(self.layer_stride)
        layer_repeats = copy.deepcopy(self.layer_repeats)
        
        graphs = copy.deepcopy(self.graphs)
        
        final_concat_layer = self.final_concat_layer
        mutation_actions = random_subset(mutation_types)

        
        if 'cell_mutation' in mutation_actions:
            new_graph = random.randint(0, 31)
            graphs[mutation_idx] = new_graph
        
        if 'channels_mutation' in mutation_actions:
            multi = random.choice([2.5, 2, 1.5, 1.25, 1, 1/1.25, 1/1.5, 1/2, 1/2.5])
            start = layer_channels[mutation_idx]
            target = layer_channels[mutation_idx] * multi
            layer_channels[mutation_idx] = clamp(target, channel_base, channel_base*channel_range)
            layer_channels[mutation_idx] = int(layer_channels[mutation_idx] - layer_channels[mutation_idx] % 8)
            print('mutated channel from{} to  {} after mutate multi {}'.format(start, layer_channels[mutation_idx], multi))
        
        if 'stride_mutation' in mutation_actions:
            layer_stride[mutation_idx] = random.choice(stride_option)
            
        
        if 'repeats_mutation' in mutation_actions:
            mutate = random.choice([0, 1, 2, -1, -2])
            ori = layer_repeats[mutation_idx]
            remain_budgets = 20 - (sum(layer_repeats) + mutate)
            if remain_budgets > 0:
                new_repeats=  max(0, min(layer_repeats[mutation_idx]+mutate, remain_budgets))
                
                if new_repeats == 0:
                    if len(layer_repeats) > 1:
                        layer_channels.pop(mutation_idx) 
                        layer_repeats.pop(mutation_idx) 
                        layer_stride.pop(mutation_idx) 
                        graphs.pop(mutation_idx)
                        
                elif new_repeats >4 :
                    layer_repeats.insert(mutation_idx+1, new_repeats-4)
                    layer_channels.insert(mutation_idx+1, layer_channels[mutation_idx]) 
                    layer_stride.insert(mutation_idx+1, layer_stride[mutation_idx]) 
                    graphs.insert(mutation_idx+1, graphs[mutation_idx]) 
                else:
                    layer_repeats[mutation_idx] = new_repeats
        
                print('mutation repeat changed from {} to {}'.format(ori, new_repeats))
        
        if 'last_concat_mutation' in mutation_actions:
            multi = random.choice([2.5, 2, 1.5, 1.25, 1, 1/1.25, 1/1.5, 1/2, 1/2.5])
            start = final_concat_layer
            target = final_concat_layer * multi
            final_concat_layer = clamp(target, channel_base, channel_base*channel_range)
            final_concat_layer = int(final_concat_layer - final_concat_layer % 8)
            print('mutated channel from{} to  {} after mutate multi {}'.format(start, final_concat_layer, multi))
        
            
            
             
        return layer_channels, layer_stride, layer_repeats, graphs, final_concat_layer