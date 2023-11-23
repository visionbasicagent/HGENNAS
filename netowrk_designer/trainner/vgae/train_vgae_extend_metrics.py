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
import torch.nn.functional as F
import network_designer.utils as utils
from network_designer.design_space.blox.design_space import BloxDesignSpace
from network_designer.design_space.nasbench201.design_space import NB201LikeDesignSpace
from network_designer.design_space.ZenNAS.design_space import ZenNASDesignSpace

import math
#load model 
from network_designer.models.vgae import VGAE
from sklearn.model_selection import train_test_split

import argparse
#from pytorch_metric_learning import losses
from pytorch_metric_learning.losses import SoftTripleLoss, TripletMarginLoss
from pytorch_metric_learning.regularizers import SparseCentersRegularizer
from pytorch_metric_learning.miners import MultiSimilarityMiner

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':4g', disp_avg=True):
        self.name = name
        self.fmt = fmt
        self.disp_avg = disp_avg
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}{val' + self.fmt + '}'
        fmtstr = fmtstr.format(name=self.name, val=self.val)
        if self.disp_avg:
            fmtstr += '({avg' + self.fmt + '})'
            fmtstr = fmtstr.format(avg=self.avg)
        return fmtstr.format(**self.__dict__)
    
def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # print(output.size())
    # print(target.size())
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
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
    def __init__(self, w_adj=1.0, w_ops=1.0, w_kl=0.1, loss_adj=nn.BCEWithLogitsLoss, loss_ops=nn.BCEWithLogitsLoss, num_features=6, n_nodes=9):
        super().__init__()
        self.w_ops = w_ops
        self.w_adj = w_adj
        self.w_kl = w_kl
        pos_weight= torch.tensor([6.0]).cuda()
        self.loss_ops = loss_ops(pos_weight=pos_weight).cuda()
        self.loss_adj = loss_adj().cuda()

    def __call__(self, inputs, targets, mu, logvar):
        adj_recon, ops_recon = inputs[0], inputs[1]
        adj, ops = targets[0], targets[1]
        b,l = mu.size()
        loss_adj = self.loss_adj(adj_recon, adj)
        loss_ops = self.loss_ops(ops_recon, ops)
        loss = self.w_ops * loss_ops + self.w_adj * loss_adj
        KLD =  -0.5 * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), dim=1))
        KLD /= b  # normalize

        return loss + (self.w_kl * KLD)
    
def train_autoencoder(m1, trainloader, epoch_step, epoch, num_epochs, lr, optimizer, criterion, triplet_loss, cond_loss=nn.CrossEntropyLoss()):
    m1.train()
    train_loss = 0
    top1 = AverageMeter('Acc@1 ', ':6.2f')
    top5 = AverageMeter('Acc@5 ', ':6.2f')
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
        
        adj_recon, ops_recon, mu, logvar, z, pred_cond = m1.forward_decoder((adj, ops))
        rec_loss = criterion((adj_recon, ops_recon), (adj, ops), mu, logvar)
        
        un_adj_recon, un_ops_recon, un_mu, un_logvar, _, _ = m1.forward_decoder((un_adj, un_ops))
        un_rec_loss = criterion((un_adj_recon, un_ops_recon), (un_adj, un_ops ), un_mu, un_logvar)

        one_hot_cond = F.one_hot(cond.flatten(start_dim=1), num_classes=64)
        #print(one_hot_cond.size())
        #t_indice = miner(z.flatten(start_dim=1), cond.flatten())
        t_loss = triplet_loss(z.flatten(start_dim=1), cond.flatten())
        pred_loss = cond_loss(pred_cond, one_hot_cond.squeeze(1).to(torch.double))
        loss = rec_loss + t_loss + un_rec_loss + pred_loss
        

        loss.backward()
        
        cond_size = int(cond.size(0))
        acc1, acc5 = accuracy(pred_cond, cond, topk=(1, 5))
        top1.update(float(acc1[0]), cond_size)
        top5.update(float(acc5[0]), cond_size)
        nn.utils.clip_grad_norm_(m1.parameters(), 5)
        optimizer.step() 
        
        try:
            loss.data[0]
        except IndexError:
            loss.data = torch.reshape(loss.data, (1,))
            
        train_loss += loss.data[0]
        
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\tLoss: %.4f \t Labelled_Rec_Loss: %.4f \tTriplet_loss %.4f \t UnLabelled_Rec_Loss %.4f \t Pred_loss %.4f \t Pred_acc_1 %.4f \t Pred_acc_5 %.4f'
                         % (epoch, num_epochs, batch_idx+1,
                            (epoch_step)+1, loss.data[0], rec_loss, t_loss, un_rec_loss, pred_loss, acc1[0], acc5[0]))
        sys.stdout.flush()
        
    return train_loss
        
def test_autoencoder(m1, test_loder, test_set, epoch, criterion, triplet_loss, cond_loss=nn.CrossEntropyLoss()):
    loss = 0.
    all_pred_loss = 0.
    m1.eval()
    correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave = 0, 0, 0, 0
    
    
    top1 = AverageMeter('Acc@1 ', ':6.2f')
    top5 = AverageMeter('Acc@5 ', ':6.2f')
    for batch_idx,  ((adj, ops), cond) in enumerate(tqdm(test_loder)):
        torch.set_grad_enabled(False)
        adj = adj.cuda()
        ops = ops.cuda()
        cond = cond.cuda()
        one_hot_cond = F.one_hot(cond, num_classes=64)
        with torch.no_grad():
            adj_recon, ops_recon, mu, logvar, z, pred_cond = m1.forward_decoder((adj, ops))
            rec_loss = criterion((adj_recon, ops_recon), (adj, ops), mu, logvar)
            pred_loss = cond_loss(pred_cond, one_hot_cond.squeeze(1).to(torch.double))
            loss += rec_loss
            loss += pred_loss
            all_pred_loss += pred_loss
            #loss += dist_loss(mu.flatten(start_dim=1), cond)
            if batch_idx == 700:
                print(adj.int())
                print( (torch.nn.Sigmoid()(adj_recon) > 0.5).int())
                print(ops.int())
                print( (torch.nn.Sigmoid()(ops_recon) > 0.5).int())
                print(torch.argmax(pred_cond, dim=-1))
                print(cond)

            acc1, acc5 = accuracy(pred_cond, cond, topk=(1, 5))

            cond_size = int(cond.size(0))
            top1.update(float(acc1[0]), cond_size)
            top5.update(float(acc5[0]), cond_size)
                
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
    #pred_acc = correct / len(test_set)
    print(f'/n')
    print(f'Average loss of test set {epoch}: {avg_loss} | Op Acc:{correct_ops_ave} | Mean Correct Adj:{mean_correct_adj_ave} | Mean FP Adj: {mean_false_positive_adj_ave} | Adj Acc:{correct_adj_ave} | Pred Acc1:{top1.avg}| Pred Acc5:{top5.avg} ')
    
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
    #dist_loss = CustomTripletMarginLoss()
    
    la = 20
    gamma =  0.1
    centers_per_class = 8
    embedding_size = 256
    num_classes = 64
    reg_weight = 0.2
    margin = 0.01
    
    
    weight_regularizer = SparseCentersRegularizer(
        64, centers_per_class
    )
  
    triplet_loss = SoftTripleLoss(
        num_classes,
        embedding_size,
        centers_per_class=centers_per_class,
        la=la,
        gamma=gamma,
        margin=margin,
        weight_regularizer=weight_regularizer,
        weight_reg_weight=reg_weight
    )



    # triplet_loss = TripletMarginLoss(margin=0.1)
    # miner = MultiSimilarityMiner(epsilon=0.1)
    train_iter = train_budget

    epoch_step = (len(trainloader))+1
    epoch_num = train_iter//epoch_step
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epoch_num), eta_min=0)
    
    es = utils.EarlyStopping(mode='min', patience=math.floor(epoch_num*0.1) if math.floor(epoch_num*0.1)>35 else 35)
    for epoch in range(1, 1+epoch_num):
        lr = scheduler.get_last_lr()[0]
        start_time = time.time()
        train_loss = train_autoencoder(m1, trainloader, epoch_step, epoch, epoch_num, lr, optimizer, criterion=criterion, triplet_loss=triplet_loss)
        
        correct_ops_ave, correct_adj_ave, avg_loss = test_autoencoder(m1, testloader, testdataset, epoch, criterion=criterion, triplet_loss=triplet_loss)
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
                torch.save(best_vgae_weight, os.path.join(save_path, 'm1_dist_{}_{}_{}_{}.pt'.format(save_label, num_layers, num_hidden, num_latent)))
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
                                labels=testset['gmm_labels'].values)

    labelled_train_dataset = utils.GraphDataset(adj_matrix=label_dataset['adj_matrix'].values, 
                                    ops_feature=label_dataset['ops_features'].values, 
                                    labels=label_dataset['gmm_labels'].values)
    
    train_loader = torch.utils.data.DataLoader(unlabelled_train_dataset,batch_size=batch_size, shuffle=True,collate_fn=custom_collate, drop_last=True)           
    train_label_loader = torch.utils.data.DataLoader(labelled_train_dataset,batch_size=batch_size, shuffle=True,collate_fn=custom_collate, drop_last=True)       
                           
    test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,collate_fn=custom_collate, drop_last=True)     

    mixed_trainloader = utils.MixedDataLoader(train_label_loader, train_loader)   
      
    m1 = VGAE(num_features=num_features, num_layers=num_layers, num_hidden=num_hidden, num_latent=num_latent, num_node=n_nodes, num_cond=num_condition)
    m1 = m1.cuda()
    
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
parser.add_argument('--num_condition', type=int, default=64)
#train parameters
parser.add_argument('--train_budget', type=int, default=550000)
parser.add_argument('--lr', type=float ,default=0.00035)

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
    
    _, test_dataset = train_test_split(label_dataset, test_size=0.05, random_state=0)

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