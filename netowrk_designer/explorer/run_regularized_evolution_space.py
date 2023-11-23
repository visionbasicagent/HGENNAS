import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
import logging
import torch
import pickle
import argparse
import copy
import numpy as np
import random
import collections
import pandas as pd
from tqdm import tqdm
import math
import json
from network_designer.dataloader import define_dataloader
from alethiometer import get_cifar_dataloaders
import hashlib

from network_designer.design_space.extend.masterspace import MasterSpace
from network_designer.design_space.extend.mastermodel import MasterModel
from network_designer.design_space.extend.operations import SearchSpaceNames
from network_designer.graph_generator.generator import GraphGenerator
from network_designer.explorer.utils import *
from tqdm import tqdm
search_space = SearchSpaceNames['nf_graph']
from ptflops import get_model_complexity_info
parser = argparse.ArgumentParser()

parser.add_argument('--design_space', default="masternet", type=str, help='specifies the benchmark')
parser.add_argument('--space_n_iters', default=3000, type=int, help='number of iterations for optimization method')
parser.add_argument('--n_iters', default=20000, type=int, help='number of iterations for optimization method')
parser.add_argument('--save_path', default="../../experiments", type=str,
                    help='specifies the path where the results will be saved')
parser.add_argument('--initial_sample_size', default=5000, type=int, help='random sample baseline')
parser.add_argument('--pop_size', default=512, type=int, help='space population size')
parser.add_argument('--arch_pop_size', default=512, type=int, help='arch population size')
parser.add_argument('--sample_size', default=10, type=int, help='sample_size')
parser.add_argument('--seed', default=1, type=int, help='random seeds range from 0 to 1')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--budget_model_size', type=float, default=None, help='budget of model size ( number of parameters), e.g., 1e6 means 1M params')
parser.add_argument('--budget_flops', type=float, default=None, help='budget of flops, e.g. , 1.8e6 means 1.8 GFLOPS')
parser.add_argument('--mutate_type', type=str, default='GMM_GUIDE', choices=['RESAMPLE', 'CELLMUTATE', 'GMM_GUIDE'], help='choose mutate type')
parser.add_argument('--GMM_path', type=str, default="../../ckpt/gmm.pkl")
parser.add_argument('--GVAE_path', type=str, default="../../experiments/extend/step_1/m1_dist_1.0_4_512_256.pt")
parser.add_argument('--CCNF_path', type=str, default="../../experiments/extend/cnf_.pt")
args = parser.parse_args()

def get_initial_space():
  
    initial_graphs = []
    for i in range(4):
        initial_graphs.append(21)

    inital_layer_channels = [8,16,32,64]
    
    inital_layer_stride =[2, 2, 2,2] 
    inital_layer_repeats = [1,1,1,1]
    
    inital_final_concat_layer = 128
    
    return inital_layer_channels, inital_layer_stride, inital_layer_repeats, initial_graphs, inital_final_concat_layer

def space_exploration(args, gpu, graph_generator=None):
    population = collections.deque()
    history = []
    parent_history = []
    histroy_count = 0
    if args.dataset == 'imagenet':
        stem_stride = 2        
        batch_size = 64
    else:
        stem_stride = 1
        batch_size = 64
    dataloader, n_classes, input_size = define_dataloader(dataset=args.dataset, dataset_path='../data', batch_size=batch_size)

    #logging.info(n_classes)

    
    layer_channels, layer_stride, layer_repeats, graphs, final_concat_layer = get_initial_space()
    space = MasterSpace(layer_channels, 
                        layer_stride, 
                        layer_repeats, 
                        final_concat_layer, 
                        graphs, 
                        n_classes=n_classes,
                        evaluate_size=1, 
                        stem_stride=stem_stride,
                        dataset=args.dataset,)

   
    space.evaluate_space(graph_generator, input_size, gpu, dataloader)
    space.get_average_population_statistics()
    histroy_count+=1
    print((space.avg_param, space.avg_flops, space.avg_zc_1), histroy_count)

    population.append(space)

    #shrink popluation size for quick generations
    if len(population) > args.pop_size:
        population = random.sample(population, args.pop_size)
    else:
        population = list(population)
    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    while histroy_count < args.space_n_iters:
        # # Sample randomly chosen models from the current population.
        print('Evolution on {} iters'.format(len(history)))
        # sample  = random.sample(population, args.sample_size)

        # # The parent is the best model in the sample.
        # parent = max(sample, key=lambda i: i.zc_sum)
        # print(parent.zc_sum)
        parent  = random.sample(population, 1)[0]

        # Create the child model and store it.
        layer_channels, layer_stride, layer_repeats, graphs, final_concact_layer  = parent.mutate()
        # print(layer_channels)
        # print(layer_stride)
        space = MasterSpace(layer_channels, 
                            layer_stride,
                            layer_repeats,
                            final_concact_layer, 
                            graphs,evaluate_size=1, 
                            n_classes=n_classes, 
                            stem_stride=stem_stride)

        space.evaluate_space(graph_generator, input_size, gpu, dataloader)
        space.get_average_population_statistics()
        print((space.avg_param, space.avg_flops, space.avg_zc_1), histroy_count)
        
        #hash_value = generate_network_hash(layer_channels, layer_stride, layer_repeats, adj_matrixs, opss)

        if args.budget_model_size is not None:
            if args.budget_model_size < space.avg_param:
                continue      

        if args.budget_flops is not None:
            if args.budget_flops < space.avg_flops:
                continue


        population.append(space)
        history.append(space)
        parent_history.append(parent)
        histroy_count+=1
    
        
        if len(population) > args.pop_size:
            # Remove the lowest score one
            lowest = min(population, key=lambda i: i.avg_zc_1)
            tmp_idx = population.index(lowest)
            population.pop(tmp_idx)
            
        # #checkpoint for every 10000 steps
        if len(history)%100 == 0:
            best_ea_sample = max(population, key=lambda i: i.avg_zc_1)
            logging.info('REA find best space at iter {} is zc: {}'.format(histroy_count, best_ea_sample.avg_zc_1))
            #history = []
    
    cache_space_record(history, args.save_path, histroy_count, args.mutate_type, args.budget_model_size, args.budget_flops, dataset=args.dataset)
    cache_space_record(parent_history, args.save_path, histroy_count, args.mutate_type, args.budget_model_size, args.budget_flops, dataset=args.dataset,post_fix='parent')
        
    best_ea_sample = max(population, key=lambda i: i.avg_zc_1)
    print('REA find best space at zc: {}'.format(best_ea_sample.avg_zc_1))
    return history, population, histroy_count*1
               
def space_exploitation(args, gpu, init_space, graph_generator=None, model_evaluated=0):
    population = collections.deque()
    history = []
    parent_history = []
    histroy_count = model_evaluated
    dataloader, n_classes, input_size = define_dataloader(dataset=args.dataset, dataset_path='../data')
    if args.dataset == 'imagenet':
        stem_stride = 2
    else:
        stem_stride = 1
    # #get init arch from found space
    # layer_channels = init_space.population[0].layer_channels
    # layer_stride = init_space.population[0].layer_stride
    # layer_repeats = init_space.population[0].layer_repeats
    # adjs = init_space.population[0].adjs
    # opss = init_space.population[0].opss

    # model = MasterModel(layer_channels, layer_stride, layer_repeats, adjs=adjs, opss=opss, n_classes=n_classes)
    
    # macs, params = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
    # flops = float(macs) * 2
    # params = float(params)

    # model.get_zc_info(dataloader, gpu=gpu, micro=False, input_size=input_size)
    # model.flops = flops
    # model.params = params
    # print(model.score, histroy_count)

    for space in init_space:
        population.append(space.population[0])


    #shrink popluation size for quick generations
    if len(population) > args.arch_pop_size:
        population = random.sample(population, args.arch_pop_size)
    else:
        population = list(population)
    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    while histroy_count < args.n_iters:
        # # Sample randomly chosen models from the current population.
        print('Evolution on {} iters'.format(len(history)))
        # sample  = random.sample(population, args.sample_size)

        # # The parent is the best model in the sample.
        # parent = max(sample, key=lambda i: i.zc_sum)
        # print(parent.zc_sum)
        parent  = random.sample(population, 1)[0]

        # Create the child model and store it.
        layer_channels, layer_stride, layer_repeats, adj_matrixs, opss, final_concact_layer  = parent.mutate(generator=graph_generator, cell_mutate_strategy=args.mutate_type)
        # print(layer_channels)
        # print(layer_stride)

        model = MasterModel(layer_channels, 
                            layer_stride, 
                            layer_repeats, 
                            final_concact_layer, 
                            adjs=adj_matrixs, 
                            opss=opss, 
                            num_classes=n_classes, 
                            stem_stride=stem_stride,
                            dataset=args.dataset).to(device=gpu)
    
        #if network is not valid to run skip and continue
        try:
            macs, params = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
            flops = float(macs) * 2
            params = float(params)
        except:
            continue
        
        if args.budget_model_size is not None:
            if args.budget_model_size < params:
                continue      

        if args.budget_flops is not None:
            if args.budget_flops < flops:
                continue
        try:

            model.get_zc_info(dataloader, gpu=gpu, micro=False, input_size=input_size)
            model.flops = flops
            model.params = params
            print(parent.score, model.score, histroy_count)
            if math.isnan(model.score[0]):
                continue
            population.append(model)
            history.append(model)
            parent_history.append(parent)
            histroy_count+=1
        except:
            continue
        
        if len(population) > args.arch_pop_size:
            # Remove the lowest score one
            lowest = min(population, key=lambda i: i.score[0])
            tmp_idx = population.index(lowest)
            population.pop(tmp_idx)
            
        #checkpoint for every 10000 steps
        if len(history)%100 == 0:
            best_ea_sample = max(population, key=lambda i: i.score[0])
            logging.info('REA find best model at iter {} is zc: {}'.format(len(history), best_ea_sample.score[0]))

    cache_arch_record(history, args.save_path, histroy_count, args.mutate_type, args.budget_model_size, args.budget_flops,dataset=args.dataset,folder='masterspace')
    cache_arch_record(parent_history, args.save_path, histroy_count, args.mutate_type, args.budget_model_size, args.budget_flops, dataset=args.dataset, post_fix='parent', folder='masterspace')
        
    best_ea_sample = max(population, key=lambda i: i.score[0])
    print('REA find best model at zc: {}'.format(best_ea_sample.score[0]))
    return history, best_ea_sample


if __name__  == '__main__':

    model_devices = torch.device("cuda:0")
    generator_devices = torch.device("cuda:1")
    dataloader, n_classes, input_size = define_dataloader(dataset=args.dataset, dataset_path='../data')
    if args.budget_model_size:
        create_logging(log_filename = os.path.join(args.save_path, 'masterspace/param_{}_{}_{}.txt'.format(args.mutate_type, args.budget_model_size, args.dataset)))
    if args.budget_flops:
        create_logging(log_filename = os.path.join(args.save_path, 'masterspace/flops_{}_{}_{}.txt'.format(args.mutate_type, args.budget_flops, args.dataset)))
        

    g_gen = GraphGenerator(gvae_ckpt=args.GVAE_path, gmm_ckpt=args.GMM_path, gnf_ckpt=args.CCNF_path,gpu=generator_devices)
    g_gen.load_components()

    print(args.n_iters)
    for seed in range(args.seed):
        np.random.seed(seed)
        random.seed(seed)

        history, best_ea_space, model_evaluated = space_exploration(args, model_devices, graph_generator=g_gen)        
        space_exploitation(args, model_devices, best_ea_space, graph_generator=g_gen, model_evaluated=model_evaluated)
