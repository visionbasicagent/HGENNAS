import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
import torch
import pickle
import argparse
import numpy as np
import random
import collections
import pandas as pd
from tqdm import tqdm

from network_designer.design_space.nasbench201.design_space import NB201LikeDesignSpace
from network_designer.design_space.extend.design_space import ExtendDesignSpace
from network_designer.design_space.core.search_space import calc_graph_hash
from network_designer.design_space.extend.search_space import calc_graph_hash as calc_graph_hash_extend
from network_designer.dataloader import define_dataloader

from network_designer.design_space.nasbench201.model import NB201LikeModel
from network_designer.design_space.extend.model import ExtendModel
from ptflops import get_model_complexity_info
parser = argparse.ArgumentParser()

parser.add_argument('--design_space', default="extend", type=str, help='specifies the benchmark')
parser.add_argument('--n_iters', default=2000, type=int, help='number of iterations for optimization method')
parser.add_argument('--save_path', default="../../experiments", type=str,
                    help='specifies the path where the results will be saved')
parser.add_argument('--initial_sample_size', default=500, type=int, help='random sample baseline')
parser.add_argument('--pop_size', default=100, type=int, help='population size')
parser.add_argument('--sample_size', default=10, type=int, help='sample_size')
parser.add_argument('--seed', default=1, type=int, help='random seeds range from 0 to 1')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--budget_model_size', type=float, default=1e6, help='budget of model size ( number of parameters), e.g., 1e6 means 1M params')
args = parser.parse_args()

input_size = (3, 32, 32)

def count_parameters(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)

def rugularized_evolution(args, initial_adj_list, initial_ops_list, design_space, gpu):
    print(design_space)
    population = collections.deque()
    history = []
    hash_mem = []
    

    dataloader, n_classes = define_dataloader(dataset=args.dataset, dataset_path='../data')

    
    # Initialize the population with random models.
    for adj, ops in tqdm(zip(initial_adj_list, initial_ops_list), total=len(initial_ops_list)):
        o = ops.astype(int)
        a = adj.astype(int)
 
        if args.design_space == 'nasbench201':
            hash = calc_graph_hash(a, o)
            model = NB201LikeModel(a, o, design_space=design_space)
        elif args.design_space == 'extend':
            hash = calc_graph_hash_extend(a, o)
            model = ExtendModel(a, o, design_space=design_space)
            #print(model.model)

        model.hash = hash
        
        try:
            if args.budget_model_size is not None:
                params = count_parameters(model)
                if args.budget_model_size < float(params):
                    continue       
        except:
            continue
        
        #print(hash_mem)
        if hash not in hash_mem :
            model.get_zc_info(dataloader, gpu=gpu)
            print(model.score)
            population.append(model)
            history.append(model)
            hash_mem.append(hash)
    
    best_random_sample = max(population, key=lambda i: (i.score[0]*i.score[1]))
    
    print('random sampled baseline find model at zc: {}'.format(best_random_sample.score[0]*best_random_sample.score[1]))
    
    #shrink popluation size for quick generations
    population = random.sample(population, args.pop_size)
    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    while len(history) < args.n_iters:
        # # Sample randomly chosen models from the current population.
        print('Evolution on {} iters'.format(len(history)))
        # sample  = random.sample(population, args.sample_size)

        # # The parent is the best model in the sample.
        # parent = max(sample, key=lambda i: i.zc_sum)
        # print(parent.zc_sum)
        parent  = random.sample(population, 1)[0]

        # Create the child model and store it.
        adj, ops  = design_space.mutate_arch(parent.adj, parent.ops)
        
        #no sample as required
        if adj is None:
            continue
        
        o = ops.astype(int)
        a = adj.astype(int)


        if args.design_space == 'nasbench201':
            hash = calc_graph_hash(a, o)
            model = NB201LikeModel(a, o, design_space=design_space)
        elif args.design_space == 'extend':
            hash = calc_graph_hash_extend(a, o)
            model = ExtendModel(a, o, design_space=design_space)
            #print(model.model)
        
        try:
            if args.budget_model_size is not None:
                params = count_parameters(model)
                if args.budget_model_size < float(params):
                    continue       
        except:
            continue
            
        try:
            model.get_zc_info(dataloader, gpu=gpu)
            print(model.score)
            population.append(model)
        except:
            continue
        
        # Remove the lowest score one
        lowest = min(population, key=lambda i: i.score[0]*i.score[1])
        tmp_idx = population.index(lowest)
        population.pop(tmp_idx)
        
        if hash not in hash_mem :
            history.append(model)
            hash_mem.append(hash)
    
    best_ea_sample = max(population, key=lambda i: i.score[0]*i.score[1])
    print('REA find best model at zc: {}'.format(best_ea_sample.score[0]*best_ea_sample.score[1]))
    return history, hash_mem, best_random_sample, best_ea_sample
               
def pick_gpu_lowest_memory():
    import gpustat
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.memory_used)/float(gpu.memory_total), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    return bestGPU

if __name__ == '__main__':
    gpu = pick_gpu_lowest_memory()
    torch.cuda.set_device(gpu)
    
    adj_matrixs_list = []
    ops_features_list = []
    zc=[]
    hash_mem = []
    
    for seed in range(args.seed):
        np.random.seed(seed)
        random.seed(seed)
        if args.design_space == 'nasbench201':
            design_space = NB201LikeDesignSpace(args=args)
        elif args.design_space == 'extend':
            design_space = ExtendDesignSpace(args=args)
        
        print(design_space)

        adj_matrixs, ops_features, _ = design_space.sample_unique_graphs(sample_size=args.initial_sample_size, seed=seed)
        
        history, hash_list, best_random_sample, best_ea_sample = rugularized_evolution(args, adj_matrixs, ops_features, design_space, gpu)
        
        for his, hash in zip(history, hash_list):
            if hash not in hash_mem:
                adj_matrixs_list.append(his.adj)
                ops_features_list.append(his.ops)
                zc.append(his.score[0]*his.score[1])
                hash_mem.append(hash)
        
        
    df = pd.DataFrame(
    {
        'adj_matrix': adj_matrixs_list,
        'ops_features': ops_features_list,
        'zc_score': zc
    })

    save_path = os.path.join(args.save_path, args.design_space)
    os.makedirs(save_path, exist_ok=True)
    
    print(df)
    df.to_pickle(os.path.join(save_path, 'ea_graph_with_vec_{}.pkl'.format(args.budget_model_size)))  
    
    with open(os.path.join(save_path, 'best_random_sample.pkl'), "wb") as f: # "wb" because we want to write in binary mode
        pickle.dump((best_random_sample.adj,best_random_sample.ops), f)
        
        
    with open(os.path.join(save_path, 'best_ea_sample_{}.pkl'.format(args.budget_model_size)), "wb") as f: # "wb" because we want to write in binary mode
        pickle.dump((best_ea_sample.adj,best_ea_sample.ops), f)