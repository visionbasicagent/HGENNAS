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

import argparse
import numpy as np
from math import log, pi

from network_designer.models.vgae import Conditional_VGAE
from torch.distributions import Normal
from torch.distributions.independent import Independent

from sklearn.model_selection import train_test_split

def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2


# def nll_loss(z, log_det):
#     log_probs = -0.5 * (z ** 2 + 2 * log_det + log(2 * pi))
#     return -torch.mean(log_probs)

class GraphDataset(Dataset):
 
    def __init__(self,adj_matrix, ops_feature, condition):
        
        adj_matrix = np.array([[np.array(i)] for i in adj_matrix])
        ops_feature = np.array([[np.array(i)] for i in ops_feature])
        condition = np.array([[np.array(i)] for i in condition])

        
        self.adj_matrix=torch.tensor(adj_matrix,dtype=torch.double)
        self.ops_feature=torch.tensor(ops_feature,dtype=torch.double)
        self.condition = torch.tensor(condition, dtype=torch.double)

    def __len__(self):
        return len(self.condition)
   
    def __getitem__(self,idx):


        x = self.adj_matrix[idx]
        y = self.ops_feature[idx]
        
        return (x.squeeze(0), y.squeeze(0)), self.condition[idx]

def train_rev(m2, gnf, trainloader, trainset, epoch, num_epochs, batch_size, lr, use_cuda, optimizer):

    m2.eval()
    gnf.train()
    train_loss = 0
    
    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, lr))
    for batch_idx, (inputs, cond) in enumerate(trainloader):
        optimizer.zero_grad()
        #inputs, targets = Variable(inputs), Variable(targets)
        inputs = (inputs[0].cuda(), inputs[1].cuda())
        batch_size = inputs[0].size()[0]
        cond = cond.cuda().squeeze(1)
        #print(ref_zc.size())
        if batch_idx == 1:
            print(cond)
        with torch.no_grad():
            
            z = m2((inputs[0], inputs[1], cond))
         
            embs = z.detach()
            
            #print(embs.size())
            b, l = embs.size()

        z, dlogp = gnf(embs, cond, torch.zeros(b, 1).to(embs))
        #log_pz = standard_normal_logprob(z).view(b, -1).sum(1, keepdim=True)
        #log_pz = Independent(Normal(torch.zeros(n, l), torch.zeros(n, l)), 1).log_prob(z).view(batch_size, -1).sum(1, keepdim=True)
        log_pz = Normal(0, 1).log_prob(z).sum(-1)
        
        #delta_log_pz = dlogp.view(b, 1).sum(1)
        log_px = log_pz - dlogp

        # # #print(log_pz.size())
        # # #delta_log_pz = dlogp.view(batch_size, 1)
        # # #dlogp = dlogp.view(batch_size, n, 1).sum(1)
        # # log_px = log_pz - dlogp
        # # nll = -log_px
        # # nll = torch.clamp(nll, min=0).mean()
        # # #nll = nll.mean()
        # # loss = nll
        nll = -log_px
        nll = torch.clamp(nll, min=0).mean()
        loss = nll.mean()

        
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
parser.add_argument('--m2_model_path', type=str, default="../../experiments/DEMO/step_1/checkpoint/vgae_1.0.pt")

#nf model parameters
parser.add_argument('--num_cond', type=int, default=17, help='zc-vector length')

#daraset
parser.add_argument('--train_set_path', type=str, default="../../experiments/DEMO/step_0/nb201_like_random_clusters_dataset.pkl")
parser.add_argument('--batch_size', type=int, default=16)

#train parameters
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--lr', type=float ,default=1e-3)

#exp 
parser.add_argument('--exp', type=str, default='Test')
parser.add_argument('--save', type=str, default='')
args = parser.parse_args()

if __name__ == '__main__':
    save_path = "{}{}".format(args.save, args.exp)
    os.makedirs(save_path, exist_ok=True)
    f = open(args.train_set_path, 'rb')
    data = pickle.load(f)
    dataset = pd.DataFrame(data)
    
     #set devices 
    if not torch.cuda.is_available():
        device = torch.device("cpu")  
    else:
        device = utils.pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
        torch.cuda.set_device(device)
    print("Using device", device)
        
    train_dataset = GraphDataset(adj_matrix=dataset['adj_matrix'].values, 
                                ops_feature=dataset['ops_features'].values, 
                                condition=dataset['conditions'].values)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16, shuffle=True, drop_last=True)
    
    gnf = GraphFlow(num_latent=args.num_latent, num_node=args.num_node)
    gnf = gnf.to(device=device).double()
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    m2 = Conditional_VGAE(num_features=args.num_features, num_layers=args.num_layers, num_hidden=args.num_hidden, num_latent=args.num_latent, num_cond=args.num_cond, num_node=args.num_node).cuda()

    # m2 = ConditionalVAE(n_feat=args.num_features, hidden_dim=args.num_hidden, latent_dim=args.num_latent, cond_dim=3, n_nodes=args.num_node).cuda()
    # m2 = m2.cuda()
    m2.load_state_dict(torch.load(args.m2_model_path))
    m2.eval().to(device=device)

    lr = args.lr
    print('|  Initial Learning Rate: ' + str(lr))
    import copy
    elapsed_time = 0
    optimizer = torch.optim.Adam(gnf.parameters(), lr=lr)
    best_loss = float('inf')
    use_cuda=True
    # def lambda_rule(ep):
    #     lr_l = 1.0 - max(0, ep - 0.5 * args.epoch) / float(0.5 * args.epoch)
    #     return lr_l
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    
    for epoch in range(1, args.epoch):
        start_time = time.time()
        # scheduler.step()
        # lr = scheduler.get_last_lr()[0]
        loss = train_rev(m2, gnf, train_loader, train_dataset, epoch, args.epoch, 16, lr, use_cuda, optimizer)
        
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
            torch.save(state, ''.join([save_point, 'cnf_m2', '.pt']))
            print('save models at epcoh: {}'.format(epoch))
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
