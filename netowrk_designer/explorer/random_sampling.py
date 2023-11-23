import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
import pickle
import argparse
import numpy as np
import random
import collections
import pandas as pd
from tqdm import tqdm
from network_designer.design_space.blox.design_space import BloxDesignSpace
from network_designer.design_space.nasbench201.design_space import NB201LikeDesignSpace
from network_designer.design_space.ZenNAS.design_space import ZenNASDesignSpace
from network_designer.design_space.nasbench201.search_space import calc_graph_hash
from network_designer.dataloader import define_dataloader

from network_designer.design_space.nasbench201.model import NB201LikeModel
from network_designer.design_space.blox.model import BloxModel
from network_designer.design_space.ZenNAS.model import ZenNASModel
from network_designer.design_space.core.utils import graph_hash_np
parser = argparse.ArgumentParser()

parser.add_argument('--design_space', default="nasbench201", type=str, help='specifies the benchmark')
parser.add_argument('--save_path', default="./experiments", type=str,
                    help='specifies the path where the results will be saved')
parser.add_argument('--sample_size', default=7000, type=int, help='random sample baseline')
parser.add_argument('--seed', default=0, type=int, help='random seeds range from 0 to 1')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--ss_path', default='', type=str, help='path to ss design space ')
args = parser.parse_args()


if __name__ == '__main__':
    
    for seed in range(args.seed):
        np.random.seed(seed)
        random.seed(seed)
        if args.design_space == 'nasbench201':
            design_space = NB201LikeDesignSpace(args=args)
            adj_matrixs, ops_features = design_space.sample_unique_graphs(sample_size=args.sample_size, seed=seed)

        elif args.design_space == 'blox':
            design_space = BloxDesignSpace(args=args)
            adj_matrixs, ops_features = design_space.sample_unique_graphs(sample_size=args.sample_size, seed=seed)

        else:
            design_space = ZenNASDesignSpace(args=args)
            adj_matrixs, ops_features = design_space.randomsample_unique_graphs(sample_size=args.sample_size, seed=seed)

        
    df = pd.DataFrame(
    {
        'adj_matrix': adj_matrixs,
        'ops_features': ops_features,
    })

    save_path = os.path.join(args.save_path, args.design_space)
    os.makedirs(save_path, exist_ok=True)
    
    df.to_pickle(os.path.join(save_path, 'vae_train_set.pkl'))  
    