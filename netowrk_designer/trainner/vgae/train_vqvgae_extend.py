import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
sys.path.insert(0, '../../../')
import time
import numpy as np
import pandas as pd
import pickle
import copy

from tqdm import tqdm
#load torch 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import network_designer.utils as utils
from network_designer.design_space.blox.design_space import BloxDesignSpace
from network_designer.design_space.nasbench201.design_space import NB201LikeDesignSpace
from network_designer.design_space.ZenNAS.design_space import ZenNASDesignSpace

import math
#load model 
from network_designer.models.vq_vgae import VQVGAE
from sklearn.model_selection import train_test_split

import argparse

from network_designer.design_space.core.search_space import SearchSpace, calc_graph_hash
ss = SearchSpace(6, 10)

def post_process_adj(adj_recon):
    adj_matrix = adj_recon.detach().cpu().numpy()
    adj_matrix = np.triu(adj_matrix, 1)[0][1:,1:]
    col, row = adj_matrix.shape
    
    #print((adj_matrix >0.5).astype(int))
    for i in range(1, col):
        #from last col:
        adj_matrix[:, col-i] = (adj_matrix[:, col-i] > 0.5).astype(int)
        #print(adj_matrix)
        #if we have incoming connection from node j, j should has at-least one incoming connection
        
        #but if a intermediate node has no output we dont help them for parent
        if i==1 or np.sum((adj_matrix[col-i,:] >0.5).astype(int)) > 0:
            for j in range(1, col-i):
                # so if node has prodecessors node j, node j should has at-least one prodecessor
                if adj_matrix[j, col-i] >0 and np.sum((adj_matrix[:,j] >0.5).astype(int)) == 0:
                    select_parent = np.argmax(adj_matrix[:,j])
                    #print('add new parent {} to node {} '.format(select_parent, j))
                    adj_matrix[select_parent, j] = 1
                
                
    adj_matrix[:, 0] = (adj_matrix[:, 0] > 0.5).astype(int) 
    
    #print(adj_matrix)
    #process loss-end
    col_sums = list(np.sum(adj_matrix, axis=0))
    row_sums = list(np.sum(adj_matrix, axis=1))
    
    need_fill_output = []
    for i ,(col, row) in enumerate(zip(col_sums[:-1], row_sums[:-1])):
        #if a node has input but no output connect it to output node
        #print(i, col, row)
        if col>0 and row == 0:
            need_fill_output.append(i)
    for o in need_fill_output:
        adj_matrix[o, -1]=1
    #print(adj_matrix) 
    return adj_matrix

def select_triplets(coordinates, margin, num_triplets):
    num_samples = len(coordinates)
    triplets = []
    
    margin = (margin * 100) / 2

    for _ in range(num_triplets):
        anchor_idx = torch.randint(0, num_samples, (1,)).item()
        positive_idx, negative_idx = None, None

        anchor_c = coordinates[anchor_idx]
        min_pos_diff = float('inf')
        max_neg_diff = float('-inf')

        for idx, c in enumerate(coordinates):
            if idx == anchor_idx:
                continue

            diff = torch.norm(anchor_c - c, p=2).item()

            if diff < margin:
                if diff < min_pos_diff:
                    min_pos_diff = diff
                    positive_idx = idx
            else:
                if diff > max_neg_diff:
                    max_neg_diff = diff
                    negative_idx = idx

        if positive_idx is not None and negative_idx is not None:
            triplets.append((anchor_idx, positive_idx, negative_idx))

    return triplets

class CustomTripletMarginLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(CustomTripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, coordinates):
        triplets = select_triplets(coordinates, self.margin, num_triplets=len(embeddings))
        loss = 0

        for anchor_idx, positive_idx, negative_idx in triplets:
            anchor_emb = embeddings[anchor_idx]
            positive_emb = embeddings[positive_idx]
            negative_emb = embeddings[negative_idx]

            pos_distance = torch.norm(anchor_emb - positive_emb, p=2)
            neg_distance = torch.norm(anchor_emb - negative_emb, p=2)

            triplet_loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0)
            loss += triplet_loss
        if len(triplets) !=0 :
            loss /= len(triplets)
        return loss
    
def custom_collate(batch):
    adj_data, ops_data, labels = [], [], []
    
    for (adj, ops), label in batch:
        adj_data.append(adj)
        ops_data.append(ops)
        labels.append(label)

    adj_data = torch.stack(adj_data)
    ops_data = torch.stack(ops_data)

    if any(label is None for label in labels):
        labels = None
    else:
        labels = torch.stack(labels)

    return (adj_data, ops_data), labels

class VAEReconstructed_Loss(object):
    def __init__(self, w_adj=1.0, w_ops=1.0, loss_adj=nn.BCEWithLogitsLoss, loss_ops=nn.BCEWithLogitsLoss, num_features=6, n_nodes=9):
        super().__init__()
        self.w_ops = w_ops
        self.w_adj = w_adj
        pos_weight= torch.tensor([6.0]).cuda()
        self.loss_ops = loss_ops(pos_weight=pos_weight).cuda()
        self.loss_adj = loss_adj().cuda()

    def __call__(self, inputs, targets):
        adj_recon, ops_recon = inputs[0], inputs[1]
        adj, ops = targets[0], targets[1]
       
        loss_adj = self.loss_adj(adj_recon, adj)
        loss_ops = self.loss_ops(ops_recon, ops)
        loss = self.w_ops * loss_ops + self.w_adj * loss_adj

        return loss 
    
def train_autoencoder(m1, trainloader, epoch_step, epoch, num_epochs, lr, optimizer, criterion, dist_loss, cond_loss=nn.MSELoss()):
    m1.train()
    train_loss = 0

    model_parameters = filter(lambda p: p.requires_grad, m1.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('|  Number of Trainable Parameters: ' + str(params))
    print('\n=> Training Epoch #%d, LR=%.5f' % (epoch, lr))
    
    for batch_idx, (((adj, ops), cond), ((un_adj, un_ops), _))in enumerate(trainloader):
    #for batch_idx, ((adj, ops), cond) in enumerate(trainloader):
       # print(cond)
        optimizer.zero_grad()
        adj = adj.cuda()
        ops = ops.cuda()
        cond = cond.cuda()
        
        un_adj = un_adj.cuda()
        un_ops = un_ops.cuda()
        
        adj_recon, ops_recon, z, pred_cond, vq_loss = m1.forward_decoder((adj, ops))
        rec_loss = criterion((adj_recon, ops_recon), (adj, ops))
        
        un_adj_recon, un_ops_recon, _, _, un_vq_loss= m1.forward_decoder((un_adj, un_ops))
        un_rec_loss = criterion((un_adj_recon, un_ops_recon), (un_adj, un_ops))
        
        triplet_loss = dist_loss(z, cond)
        pred_loss = cond_loss(pred_cond, cond)
        loss =  un_vq_loss  + vq_loss + triplet_loss + pred_loss + rec_loss + un_rec_loss
        
        #print(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(m1.parameters(), 5)
        optimizer.step() 
        
        try:
            loss.data[0]
        except IndexError:
            loss.data = torch.reshape(loss.data, (1,))
            
        train_loss += loss.data[0]
        
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\tLoss: %.4f \t Labelled_Rec_Loss: %.4f \tTriplet_loss %.4f \t UnLabelled_Rec_Loss %.4f \t Pred_loss %.4f \t VQ_Loss %.4f \t un_VQ_loss %.4ff '
                         % (epoch, num_epochs, batch_idx+1,
                            (epoch_step)+1, loss.data[0], rec_loss, triplet_loss, un_rec_loss, pred_loss, vq_loss, un_vq_loss))
        sys.stdout.flush()
        
    return train_loss
        
def test_autoencoder(m1, test_loder, test_set, epoch, criterion, dist_loss, cond_loss=nn.MSELoss()):
    loss = 0.
    all_pred_loss = 0.
    m1.eval()
    correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave = 0, 0, 0, 0
    for batch_idx,  ((adj, ops), cond) in enumerate(tqdm(test_loder)):
        torch.set_grad_enabled(False)
        adj = adj.cuda()
        ops = ops.cuda()
        cond = cond.cuda()
        with torch.no_grad():
            adj_recon, ops_recon, z, pred_cond, vq_loss = m1.forward_decoder((adj, ops))
            rec_loss = criterion((adj_recon, ops_recon), (adj, ops))
            pred_loss = cond_loss(pred_cond, cond)
            loss += rec_loss
            loss += pred_loss
            loss += vq_loss
            all_pred_loss += pred_loss
            #loss += dist_loss(mu.flatten(start_dim=1), cond)
            if batch_idx == 700:
                print(adj.int())
                print( (torch.nn.Sigmoid()(adj_recon) > 0.5).int())
                print(ops.int())
                print( (torch.nn.Sigmoid()(ops_recon) > 0.5).int())
                print(pred_cond)
                print(cond)
        torch.set_grad_enabled(True)
        

        correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj = utils.get_accuracy_extend((ops_recon, adj_recon), (ops, adj))
        
        correct_ops_ave += correct_ops
        mean_correct_adj_ave += mean_correct_adj 
        mean_false_positive_adj_ave += mean_false_positive_adj 
        correct_adj_ave += correct_adj 
    
    avg_loss = loss / len(test_set)
    avg_pred_loss = all_pred_loss / len(test_set)
    correct_ops_ave = round((correct_ops_ave / len(test_set)), 4)
    mean_correct_adj_ave = round((mean_correct_adj_ave.item() / len(test_set)) , 4)
    mean_false_positive_adj_ave = round((mean_false_positive_adj_ave.item() / len(test_set)) , 4)
    correct_adj_ave = round((correct_adj_ave / len(test_set)), 4)
    
    print(f'/n')
    print(f'Average loss of test set {epoch}: {avg_loss} | Op Acc:{correct_ops_ave} | Mean Correct Adj:{mean_correct_adj_ave} | Mean FP Adj: {mean_false_positive_adj_ave} | Adj Acc:{correct_adj_ave} | Pred loss:{avg_pred_loss} ')
    
    return correct_ops_ave, correct_adj_ave, avg_loss

def autoencoder_train(m1, trainloader, traindataset, testloader, 
                      testdataset, save=True, save_label='', save_path='',
                      batch_size=64, train_budget=250, lr=0.00035, num_features=0, n_nodes=0, num_layers=4, num_hidden=256, num_latent=128):
    print('|  Initial Learning Rate: ' + str(lr))
    optimizer = optim.AdamW(m1.parameters(), lr=lr, weight_decay=5e-4)
    highest_acc = None
    lowest_loss =None
    criterion = VAEReconstructed_Loss(w_adj=1.0, w_ops=1.0, num_features=num_features, n_nodes=n_nodes)
    # # Set the mining function
    # miner = miners.MultiSimilarityMiner(epsilon=0.1)
    dist_loss = CustomTripletMarginLoss()
    train_iter = train_budget

    epoch_step = (len(trainloader))+1
    epoch_num = train_iter//epoch_step
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epoch_num), eta_min=0)
    
    es = utils.EarlyStopping(mode='min', patience=math.floor(epoch_num*0.1) if math.floor(epoch_num*0.1)>35 else 35)
    for epoch in range(1, 1+epoch_num):
        lr = scheduler.get_last_lr()[0]
        start_time = time.time()
        train_loss = train_autoencoder(m1, trainloader, epoch_step, epoch, epoch_num, lr, optimizer, criterion=criterion, dist_loss=dist_loss)
        
        correct_ops_ave, correct_adj_ave, avg_loss = test_autoencoder(m1, testloader, testdataset, epoch, criterion=criterion, dist_loss=dist_loss)
        scheduler.step()
        
        if epoch > 35:
            if es.step(avg_loss.cpu()):
                print('Early stopping criterion is met, stop training now.')
                print(f'Best result yet {highest_acc}.. VGAE model')
                break

        if lowest_loss is None or train_loss < lowest_loss:
            lowest_loss = train_loss
            highest_acc = (correct_ops_ave, correct_adj_ave)
            best_vgae_weight = m1.state_dict()
            if save:
                torch.save(best_vgae_weight, os.path.join(save_path, 'vqvgae_dist_{}_{}_{}_{}.pt'.format(save_label, num_layers, num_hidden, num_latent)))
                print(f'Best result yet {highest_acc}.. VGAE model saved.')
            
    print(f'Best result yet {highest_acc}.. VGAE model')
  
def train_pipeline(trainset, testset, label_dataset, train_portion, num_condition=2,
                   num_features=6, num_layers=4, num_hidden=256, num_latent=32, 
                   batch_size=64, train_budget=0, lr=0.00035, save='', design_space=None, n_nodes=0, m1_path=None):
    
    
    # if args.design_space == 'nasbench201':
    #     design_space = NB201LikeDesignSpace(args=args)
    # elif args.design_space == 'nasbench101':
    #     pass
    # elif args.design_space == 'blox':
    #     design_space = BloxDesignSpace(args=args)
    # else:
    #     design_space = ZenNASDesignSpace(args=args)

        
    unlabelled_train_dataset = utils.UnlabelledGraphDataset(adj_matrix=trainset['adj_matrix'].values, 
                                ops_feature=trainset['ops_features'].values)

    # test_dataset = utils.UnlabelledGraphDataset(adj_matrix=testset['adj_matrix'].values, 
    #                                 ops_feature=testset['ops_features'].values)
    
    test_dataset = utils.GraphDataset(adj_matrix=testset['adj_matrix'].values, 
                                ops_feature=testset['ops_features'].values, 
                                labels=testset['conditions'].values)

    labelled_train_dataset = utils.GraphDataset(adj_matrix=label_dataset['adj_matrix'].values, 
                                    ops_feature=label_dataset['ops_features'].values, 
                                    labels=label_dataset['conditions'].values)
    
    train_loader = torch.utils.data.DataLoader(unlabelled_train_dataset,batch_size=batch_size, shuffle=True,collate_fn=custom_collate, drop_last=True)           
    train_label_loader = torch.utils.data.DataLoader(labelled_train_dataset,batch_size=batch_size, shuffle=True,collate_fn=custom_collate, drop_last=True)       
                           
    test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,collate_fn=custom_collate, drop_last=True)     

    mixed_trainloader = utils.MixedDataLoader(train_label_loader, train_loader)   
      
    m1 = VQVGAE(num_features=num_features, num_layers=num_layers, num_hidden=num_hidden, num_latent=num_latent, num_node=n_nodes, num_cond=num_condition)
    m1 = m1.cuda()
    
    # m1._vgae.load_state_dict(torch.load("../experiments/extend/step_1/m1_dist_1.0_4_512_256.pt"), strict=False)
    # m1._vgae.eval()
    
    autoencoder_train(m1, mixed_trainloader, unlabelled_train_dataset, 
                      test_loader, test_dataset, save_label=str(train_portion),
                      batch_size=batch_size, train_budget=train_budget, lr=lr,save_path=save, num_features=num_features, n_nodes=n_nodes, num_layers=num_layers, num_hidden=num_hidden, num_latent=num_latent)

parser = argparse.ArgumentParser("vgae")

#devices
parser.add_argument('--gpu', type=str, default='auto', help='gpu device id')
parser.add_argument('--seed', type=int, default=0, help='random seed')
#dataset
parser.add_argument('--train_portion', type=float, default=1.0, help='dataset portion for training VGAE')
parser.add_argument('--unlabel_dataset_path', type=str, default="../../exp/benchmarks/nasbench201/sampled_graph_with_vec.pkl", help='dataset path that store all the graph and its zc-vector')
parser.add_argument('--label_dataset_path', type=str, default="../../exp/benchmarks/nasbench201/sampled_graph_with_vec.pkl", help='dataset path that store all the graph and its zc-vector')
parser.add_argument('--test_dataset_path', type=str, default="")
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train loader')

#model parameters
parser.add_argument('--num_features', type=int, default=6, help='number of ops in graph nodes')
parser.add_argument('--num_layers', type=int, default=4, help='number of layers in VAE model')
parser.add_argument('--num_hidden', type=int, default=256, help='len of itermediate feature in VAE')
parser.add_argument('--num_latent', type=int, default=32, help='len of output features from encoder')
parser.add_argument('--ops_activation', type=str, default='sigmoid')
parser.add_argument('--ops_loss', type=str, default='BECLoss')
parser.add_argument('--n_nodes', type=int, default=9)
parser.add_argument('--num_condition', type=int, default=3)
#train parameters
parser.add_argument('--train_budget', type=int, default=550000)
parser.add_argument('--lr', type=float ,default=1e-3)

#exp 
parser.add_argument('--exp', type=str, default='Test')
parser.add_argument('--exp_root', type=str, default='../../../experiments')

#clustering 
parser.add_argument('--cluster_eps', type=float ,default=1.0)

parser.add_argument('--design_space', type=str, default='ImageNet800M', help='use scala ops features')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--ss_path', default='', type=str, help='path to ss design space ')


parser.add_argument('--m1_path', default="../experiments/nb201/step_1/vgae_1.0.pt", type=str, help='path to m1 ')
args = parser.parse_args()

if __name__ == '__main__':
    cudnn.benchmark = True
    cudnn.enabled = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    exp_path = "{}/{}/step_1/".format(args.exp_root, args.exp)
    os.makedirs(exp_path, exist_ok=True)
    
    #set devices 
    if not torch.cuda.is_available():
        device = torch.device("cpu")  
    else:
        device = utils.pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
        torch.cuda.set_device(device)
    print("Using device", device)

    f = open(args.unlabel_dataset_path, 'rb')
    data = pickle.load(f)
    unlabel_dataset = pd.DataFrame(data)
    
    #unlabel_dataset, test_dataset = train_test_split(unlabel_dataset, test_size=0.05, random_state=0)
    
    f = open(args.label_dataset_path, 'rb')
    data = pickle.load(f)
    label_dataset = pd.DataFrame(data)
    
    label_dataset, test_dataset = train_test_split(label_dataset, test_size=0.05, random_state=0)

    # f = open(args.test_dataset_path, 'rb')
    # data = pickle.load(f)
    # test_dataset = pd.DataFrame(data)
    
    #label_dataset = None
   
    
    num_train = len(unlabel_dataset)
    indices = list(range(num_train))
    train_pipeline(unlabel_dataset, test_dataset, label_dataset, args.train_portion, num_condition=args.num_condition,
                   num_features=args.num_features, num_layers=args.num_layers,
                   num_hidden=args.num_hidden, num_latent=args.num_latent,
                   batch_size=args.batch_size, train_budget=args.train_budget, 
                   lr=args.lr, save=exp_path, design_space=args.design_space, n_nodes=args.n_nodes, m1_path=args.m1_path)