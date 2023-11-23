import os
import sys
import random
import utils as train_utils
sys.path.insert(0, '../../')
sys.path.insert(0, '../../../')
import time
import glob
import numpy as np
import torch

import logging
import argparse
import shutil
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import json
import pickle
import pandas as pd
from network_designer.design_space.nasbench201.operations import SearchSpaceNames
from network_designer.design_space.nasbench201.model import Network

parser = argparse.ArgumentParser("latency")
parser.add_argument('--gpu', type=str, default='auto', help='gpu device id')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--exp_path', type=str, default='../../experiments/nb201-latency/', help='path to save the model')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--space_root', type=str, default='../experiments/DEMO/step_2/')
parser.add_argument('--space', type=str, default='', help="pickle file name for sampled space")
parser.add_argument('--id', type=int, default=0, help="id for architecutres in space")
parser.add_argument('--seed', type=int, default=888, help='random seed')
parser.add_argument('--search_ops_space', type=str, default='nf_graph', help='ops in search space')

args = parser.parse_args()

#### args augment
expid = args.save

def load_graph_from_pickle(space_root, space, id):
    f = open('{}/{}.pkl'.format(space_root, space), 'rb')
    data = pickle.load(f)
    dataset = pd.DataFrame(data)
    
    logging.info(dataset.iloc[id])
    adj_matrix = dataset.iloc[id]['adj_matrix']
    adj_matrix = np.triu(adj_matrix, 1)
    ops = dataset.iloc[id]['ops_features']

    
    logging.info(ops)
    
    if space == 'ref_best' or space == 'pareto' or space == 'pareto_all':
        adj_matrix = np.triu(adj_matrix, 1)
        adj_matrix = adj_matrix[1:,1:]
        ops = ops[1:]

    return adj_matrix, ops
    
args.save = '{}/{}/{}/{}-{}'.format(args.exp_path, args.space, args.seed, args.id)

os.makedirs(args.save, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
log_file = 'log.txt'
fh = logging.FileHandler(os.path.join(args.save, log_file), mode='w')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


if args.dataset == 'cifar100':
    n_classes = 100
else:
    n_classes = 10

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

def calc_stats(values):
    averages = []
    for subvalues in values:
        q25 = np.percentile(subvalues, 25)
        q75 = np.percentile(subvalues, 75)
        subvalues_filtered = list(filter(lambda x : (x >= q25) and (x <= q75), subvalues))
        averages.append(np.mean(subvalues_filtered))
    q25 = np.percentile(averages, 25)
    q75 = np.percentile(averages, 75)
    averages_filtered = list(filter(lambda x : (x >= q25) and (x <= q75), averages))
    return np.mean(averages_filtered)

def measure_latency(model, input_data, num_runs=100):
    model = model.cuda()
    model.eval()

    # Warm-up run
    with torch.no_grad():
        model(input_data)

    run_times = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Timing loop
    for _ in range(num_runs):
        start_event.record()

        with torch.no_grad():
            model(input_data)
        
        end_event.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        run_times.append(start_event.elapsed_time(end_event))

    run_times = np.array(run_times)

    logging.info(f'Average inference time over {num_runs} runs: {np.mean(run_times):.3f} ms')
    logging.info(f'Minimum inference time over {num_runs} runs: {np.min(run_times):.3f} ms')
    logging.info(f'Maximum inference time over {num_runs} runs: {np.max(run_times):.3f} ms')


def main():
    adj_matrix, ops = load_graph_from_pickle(args.space_root, args.space, args.id)
     
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
 
    gpu = train_utils.pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
    torch.cuda.set_device(gpu)
    cudnn.enabled = True
    seed_torch(args.seed)

    logging.info('gpu device = %d' % gpu)
    logging.info("args = %s", args)

    search_space = SearchSpaceNames[args.search_ops_space]
    model = Network(C=args.init_channels, N=5, num_classes=n_classes, search_space=search_space, adj_matrix=adj_matrix, ops=ops)
    
    model = model.cuda()
    
    logging.info("param size = %fMB", train_utils.count_parameters_in_MB(model))
    
    input_data = torch.rand(1, 3, 32, 32).cuda()
    
    run_stats = measure_latency(model, input_data)
    
    final_stats = calc_stats(run_stats)
    logging.info('Final_Latency:{}'.format(final_stats))

   

if __name__ == '__main__':
    main()