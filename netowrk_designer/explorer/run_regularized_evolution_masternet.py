import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')

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
from network_designer.design_space.extend.masternetwork import generate_random_arch
from network_designer.design_space.extend.mastermodel import MasterModel
from network_designer.design_space.extend.operations import SearchSpaceNames
from network_designer.graph_generator.generator import GraphGenerator
from network_designer.explorer.utils import *
from tqdm import tqdm
search_space = SearchSpaceNames['nf_graph']
from ptflops import get_model_complexity_info
parser = argparse.ArgumentParser()

parser.add_argument('--design_space', default="masternet", type=str, help='specifies the benchmark')
parser.add_argument('--n_iters', default=50000, type=int, help='number of iterations for optimization method')
parser.add_argument('--save_path', default="../../experiments", type=str,
                    help='specifies the path where the results will be saved')
parser.add_argument('--initial_sample_size', default=5000, type=int, help='random sample baseline')
parser.add_argument('--pop_size', default=512, type=int, help='population size')
parser.add_argument('--sample_size', default=10, type=int, help='sample_size')
parser.add_argument('--seed', default=1, type=int, help='random seeds range from 0 to 1')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--budget_model_size', type=float, default=None, help='budget of model size ( number of parameters), e.g., 1e6 means 1M params')
parser.add_argument('--budget_flops', type=float, default=None, help='budget of flops, e.g. , 1.8e6 means 1.8 GFLOPS')
parser.add_argument('--mutate_type', type=str, default='RESAMPLE', choices=['RESAMPLE', 'CELLMUTATE', 'GMM_GUIDE'], help='choose mutate type')
parser.add_argument('--GMM_path', type=str, default="../../experiments/extend/gmm.pkl")
parser.add_argument('--GVAE_path', type=str, default="../../experiments/extend/step_1/m1_dist_1.0_4_512_256.pt")
parser.add_argument('--CCNF_path', type=str, default="../../experiments/extend/cnf_.pt")
args = parser.parse_args()

def get_initial_arch():
    initial_adj = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0, 0, 0, 0, 1],
                    [0, 0, 1, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1]]
    initial_ops =  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
    
    initial_adjs = []
    for i in range(3):
        initial_adjs.append(np.array(copy.deepcopy(initial_adj)))

    initial_opss = []
    for i in range(3):
        initial_opss.append(np.array(copy.deepcopy(initial_ops)))
        

    inital_layer_channels = [32, 64, 64]
    
    inital_layer_stride =[1,2, 1] 
    inital_layer_repeats = [1,1, 1]

    
    return inital_layer_channels, inital_layer_stride, inital_layer_repeats, initial_adjs, initial_opss    

def rugularized_evolution(args, gpu, graph_generator=None):
    population = collections.deque()
    history = []
    parent_history = []
    histroy_count = 0
    dataloader, n_classes, input_size = define_dataloader(dataset=args.dataset, dataset_path='../data')

    layer_channels, layer_stride, layer_repeats, adj_matrixs, opss = get_initial_arch()
    model = MasterModel(layer_channels, layer_stride, layer_repeats, adjs=adj_matrixs, opss=opss, n_classes=n_classes)
    

    macs, params = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
    flops = float(macs) * 2
    params = float(params)


    model.get_zc_info(dataloader, gpu=gpu, micro=False, input_size=input_size)
    model.flops = flops
    model.params = params
    print(model.score, histroy_count)

    population.append(model)


    #shrink popluation size for quick generations
    if len(population) > args.pop_size:
        population = random.sample(population, args.pop_size)
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
        layer_channels, layer_stride, layer_repeats, adj_matrixs, opss  = parent.mutate(generator=graph_generator, cell_mutate_strategy=args.mutate_type)
        # print(layer_channels)
        # print(layer_stride)
        model = MasterModel(layer_channels, layer_stride, layer_repeats, adjs=adj_matrixs, opss=opss)

        #if network is not valid to run skip and continue
        try:
            macs, params = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
            flops = float(macs) * 2
            print(flops)
            print(args.budget_flops)
            #print(macs)
            params = float(params)
        except:
            continue
        
        print(model.model)
        #hash_value = generate_network_hash(layer_channels, layer_stride, layer_repeats, adj_matrixs, opss)

        if args.budget_model_size is not None:
            if args.budget_model_size < params:
                continue      

        if args.budget_flops is not None:
            if args.budget_flops < flops:
                continue
        # try:

        model.get_zc_info(dataloader, gpu=gpu, micro=False, input_size=input_size)
        model.flops = flops
        model.params = params
        print(parent.score, model.score, histroy_count)
        if math.isnan(model.score[0]) or math.isnan(model.score[1]):
            continue
        population.append(model)
        history.append(model)
        parent_history.append(parent)
        histroy_count+=1
        # except:
        #     continue
        
        if len(population) > args.pop_size:
            # Remove the lowest score one
            lowest = min(population, key=lambda i: i.score[0]*i.score[1])
            tmp_idx = population.index(lowest)
            population.pop(tmp_idx)
            
        #checkpoint for every 10000 steps
        if len(history)%5000 == 0:
            cache_arch_record(history, args.save_path, histroy_count, args.mutate_type, args.budget_model_size, args.budget_flops)
            cache_arch_record(parent_history, args.save_path, histroy_count, args.mutate_type, args.budget_model_size, args.budget_flops, post_fix='parent')
            best_ea_sample = max(population, key=lambda i: i.score[0]*i.score[1])
            logging.info('REA find best model at iter {} is zc: {}'.format(len(history), best_ea_sample.score[0]*best_ea_sample.score[1]))
            history = []

        
    best_ea_sample = max(population, key=lambda i: i.score[0]*i.score[1])
    print('REA find best model at zc: {}'.format(best_ea_sample.score[0]*best_ea_sample.score[1]))
    return history, best_ea_sample
               
if __name__  == '__main__':
    gpu = pick_gpu_lowest_memory()
    torch.cuda.set_device(gpu)
    dataloader, n_classes, input_size = define_dataloader(dataset=args.dataset, dataset_path='../data')
    if args.budget_model_size:
        create_logging(log_filename = os.path.join(args.save_path, 'masternet/param_{}_{}.txt'.format(args.mutate_type, args.budget_model_size)))
    if args.budget_flops:
        create_logging(log_filename = os.path.join(args.save_path, 'masternet/flops_{}_{}.txt'.format(args.mutate_type, args.budget_flops)))
        
    if args.mutate_type == 'GMM_GUIDE':
        g_gen = GraphGenerator(gvae_ckpt=args.GVAE_path, gmm_ckpt=args.GMM_path, gnf_ckpt=args.CCNF_path)
        g_gen.load_components()
    else:
        g_gen = None
    
    for seed in range(args.seed):
        np.random.seed(seed)
        random.seed(seed)

        history, best_ea_sample = rugularized_evolution(args, gpu, graph_generator=g_gen)        
