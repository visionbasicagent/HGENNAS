import os 
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
import time
from network_designer.models.cnf_modules.network import GraphFlow
import network_designer.utils as utils
import pickle
import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import joblib
import argparse
import numpy as np
from math import log, pi

from network_designer.models.vgae import VGAE
from torch.distributions import Normal

from sklearn.model_selection import train_test_split
import math

def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2

class GraphDataset(Dataset):
 
    def __init__(self,adj_matrix, ops_feature):
        
        adj_matrix = np.array([[np.array(i)] for i in adj_matrix])
        ops_feature = np.array([[np.array(i)] for i in ops_feature])
        #condition = np.array([[np.array(i)] for i in condition])

        
        self.adj_matrix=torch.tensor(adj_matrix,dtype=torch.double)
        self.ops_feature=torch.tensor(ops_feature,dtype=torch.double)
        #self.condition = torch.tensor(condition, dtype=torch.double)

    def __len__(self):
        return len(self.adj_matrix)
   
    def __getitem__(self,idx):


        x = self.adj_matrix[idx]
        y = self.ops_feature[idx]
        
        return (x.squeeze(0), y.squeeze(0))

def train_rev(m1, gnf, trainloader, trainset, epoch, num_epochs, batch_size, lr, use_cuda, optimizer):
    m1.eval()
    gnf.train()
    train_loss = 0
    # batch_count = 0 
    # batch_limits = 500
    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, lr))
    for batch_idx, inputs in enumerate(trainloader):
        optimizer.zero_grad()
        #inputs, targets = Variable(inputs), Variable(targets)
        inputs = (inputs[0].cuda(), inputs[1].cuda())
        batch_size = inputs[0].size()[0]
        #cond = cond.cuda().squeeze(1)
        #print(ref_zc.size())
       
        with torch.no_grad():
            
            z, pred_cond = m1(inputs)
            embs = z.detach()
            b, l = embs.size()
            
            # # Finding the indices of the maximum values along the dimension representing the classes
            # max_values, max_indices = torch.max(pred_cond, dim=1)

            # # Creating the one-hot encoded result
            # cls_token = torch.zeros_like(pred_cond)
            # cls_token.scatter_(1, max_indices.unsqueeze(1), 1)
  
            # #print(cls_token)


        z, dlogp = gnf(embs, pred_cond, torch.zeros(b, 1).to(embs))
        log_pz = Normal(0, 1).log_prob(z).sum(-1)
        #print(z.shape)
        #log_pz = standard_normal_logprob(z).sum(-1)
        
        dlogp = dlogp.view(b, 1).sum(1)
        log_px = log_pz - dlogp

        nll = -log_px.mean()
        loss = nll / 256 / math.log(2)


        
        dlogp_data = dlogp.mean().item()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(gnf.parameters(), 5)
        optimizer.step()

        try:
            loss.data[0]
        except IndexError:
            loss.data = torch.reshape(loss.data, (1,))
        train_loss += loss.data[0]
        
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f\t\t Loget %.4f' 
                        % (epoch, num_epochs, batch_idx+1,
                            (len(trainset)//batch_size)+1, loss.data[0], dlogp_data))
        sys.stdout.flush()
        
        # batch_count+=1
        # if batch_count > batch_limits:
        #     break

            
    return train_loss 

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

parser = argparse.ArgumentParser("cnf")

parser.add_argument('--gpu', type=str, default='auto', help='gpu device id')
parser.add_argument('--seed', type=int, default=0, help='random seed')

#encoder model parameters
parser.add_argument('--num_features', type=int, default=6, help='number of ops in graph nodes')
parser.add_argument('--num_layers', type=int, default=4, help='number of layers in VAE model')
parser.add_argument('--num_hidden', type=int, default=256, help='len of itermediate feature in VAE')
parser.add_argument('--num_latent', type=int, default=32, help='len of output features from encoder')
parser.add_argument('--num_node', type=int, default=9, help='number of node in graph')
parser.add_argument('--m1_model_path', type=str, default="../../experiments/DEMO/step_1/checkpoint/vgae_1.0.pt")

#nf model parameters
parser.add_argument('--zc_len', type=int, default=17, help='zc-vector length')

#daraset
parser.add_argument('--train_set_path', type=str, default="../../experiments/DEMO/step_0/nb201_like_random_clusters_dataset.pkl")
parser.add_argument('--unlabel_dataset_path', type=str, default="../../exp/benchmarks/nasbench201/sampled_graph_with_vec.pkl", help='dataset path that store all the graph and its zc-vector')

parser.add_argument('--batch_size', type=int, default=256)

#train parameters
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--lr', type=float ,default=1e-3)

#exp 
parser.add_argument('--exp', type=str, default='Test')
parser.add_argument('--save', type=str, default='')

# #conditions
# parser.add_argument('--zc_score_1', action='store_true', default=False, help='use zc_score_1 as nf condition')
# parser.add_argument('--zc_score_2', action='store_true', default=False, help='use zc_score_2 as nf condition')
# parser.add_argument('--param', action='store_true', default=False, help='use param as nf condition')

args = parser.parse_args()

if __name__ == '__main__':
    # slices = []
    # cond_names = []
    
    # if args.zc_score_1:
    #     slices.append(0)
    #     cond_names.append('zc_score_1')
        
    # if args.zc_score_2:
    #     slices.append(1)
    #     cond_names.append('zc_score_2')
        
    # if args.param:
    #     slices.append(2)
    #     cond_names.append('param')
        
    
    # cond_names = '_'.join(cond_names)
    # len_cond = len(slices)
    # slices = torch.tensor(slices).cuda()
    save_path = "{}{}".format(args.save, args.exp)
    # print(cond_names)
    print(save_path)
    os.makedirs(save_path, exist_ok=True)
    f = open(args.train_set_path, 'rb')
    data = pickle.load(f)
    dataset = pd.DataFrame(data)
    
    f = open(args.unlabel_dataset_path, 'rb')
    data = pickle.load(f)
    unlabel_dataset = pd.DataFrame(data)
    
    
    # = pd.concat([dataset[['adj_matrix', 'ops_features']], unlabel_dataset[['adj_matrix', 'ops_features']]], ignore_index=True)
    dataset = unlabel_dataset
     #set devices 
    if not torch.cuda.is_available():
        device = torch.device("cpu")  
    else:
        device = utils.pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
        torch.cuda.set_device(device)
    print("Using device", device)
        
    train_dataset = GraphDataset(adj_matrix=dataset['adj_matrix'].values, 
                                ops_feature=dataset['ops_features'].values)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    gnf = GraphFlow(num_latent=args.num_latent, num_node=args.num_node, num_cond=4)
    gnf = gnf.to(device=device).double()
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    m1 = VGAE(num_features=args.num_features, num_layers=args.num_layers, num_hidden=args.num_hidden, num_latent=args.num_latent, num_node=args.num_node, num_cond=4)
    m1.load_state_dict(torch.load(args.m1_model_path))
    m1.eval()
    m1 = m1.to(device=device)
    
    # # Load the model from the file
    # cls = joblib.load('../ckpt/cls.pkl')

    
    lr = args.lr
    print('|  Initial Learning Rate: ' + str(lr))
    import copy
    elapsed_time = 0
    optimizer = torch.optim.Adam(gnf.parameters(), lr=lr)
    best_loss = float('inf')
    use_cuda=True
    
    for epoch in range(1, args.epoch):
        start_time = time.time()
        # scheduler.step()
        # lr = scheduler.get_last_lr()[0]
        loss = train_rev(m1, gnf, train_loader, train_dataset, epoch, args.epoch, 16, lr, use_cuda, optimizer)
        
        print(loss)
        print(best_loss)
        if loss < best_loss:
            best_loss = loss
            state = {
                'model': gnf,
                'epoch': epoch,
            }
            # if not os.path.isdir('checkpoint'):
            #     os.mkdir('checkpoint')
            save_point = save_path
            if not os.path.isdir(save_point):
                os.mkdir(save_point)
            torch.save(state, ''.join([save_point, '/','cnf_', '.pt']))
            print('save models at epcoh: {}'.format(epoch))
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
