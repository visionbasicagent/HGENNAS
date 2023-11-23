import torch
from torch.utils.data import Dataset

import gpustat
import numpy as np
#load unsupervise clustering
from sklearn.cluster import DBSCAN

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)
                
class GraphDataset(Dataset):
 
    def __init__(self,adj_matrix, ops_feature, labels=None):
        #print(le)
        adj_matrix = np.array([[np.array(i)] for i in adj_matrix])
        ops_feature = np.array([[np.array(i)] for i in ops_feature])
        
        
        self.adj_matrix=torch.tensor(adj_matrix,dtype=torch.double)
        self.ops_feature=torch.tensor(ops_feature,dtype=torch.double)
        labels = np.array([[np.array(i)] for i in labels])
        self.labels = torch.tensor(labels, dtype=torch.double)
            
        
    def __len__(self):
        return len(self.adj_matrix)
   
    def __getitem__(self,idx):
        x = self.adj_matrix[idx]
        y = self.ops_feature[idx]
        

        z = self.labels[idx]
        return (x.squeeze(0), y.squeeze(0)), z.flatten()

    
class UnlabelledGraphDataset(Dataset):
    def __init__(self,adj_matrix, ops_feature):
        adj_matrix = np.array([[np.array(i)] for i in adj_matrix])
        ops_feature = np.array([[np.array(i)] for i in ops_feature])

        
        
        self.adj_matrix=torch.tensor(adj_matrix,dtype=torch.double)
        self.ops_feature=torch.tensor(ops_feature,dtype=torch.double)
        
    def __len__(self):
        return len(self.adj_matrix)
   
    def __getitem__(self,idx):
        x = self.adj_matrix[idx]
        y = self.ops_feature[idx]

        # x = x.squeeze(0)[1:, 1:].triu(1)
        # y = y.squeeze(0)[1:, :-1]
        
        # return (x, y)
        return (x.squeeze(0), y.squeeze(0)), None


class MixedDataLoader:
    def __init__(self, labeled_loader, unlabeled_loader):
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.labeled_iter = iter(labeled_loader)
        self.unlabeled_iter = iter(unlabeled_loader)

    def __iter__(self):
        return self

    def __next__(self):
        labeled_batch = None
        unlabeled_batch = None

        try:
            labeled_batch = next(self.labeled_iter)
        except StopIteration:
            self.labeled_iter = iter(self.labeled_loader)
            labeled_batch = next(self.labeled_iter)

        try:
            unlabeled_batch = next(self.unlabeled_iter)
        except StopIteration:
            self.unlabeled_iter = iter(self.unlabeled_loader)
            raise StopIteration

        return labeled_batch, unlabeled_batch

    def __len__(self):
        return len(self.unlabeled_loader)
    
def get_accuracy_zenNAS(inputs, targets, N, I, feature_chunks_size=None, design_space=None):
    ops_recon, adj_recon = inputs[0], inputs[1]
    ops, adj = targets[0], targets[1]
    adj_recon, adj = adj_recon.triu(1), adj.triu(1)
    mean_correct_adj = adj_recon[adj.type(torch.bool)].sum().item() / adj.sum()
    mean_false_positive_adj = adj_recon[(~adj.type(torch.bool)).triu(1)].sum().item() / (N*I*(I-1)/2.0-adj.sum())
    threshold = 0.5 # hard threshold
    adj_recon_thre = adj_recon > threshold
    correct_adj = adj_recon_thre.eq(adj.type(torch.bool)).float().triu(1).sum().item()/ (N*I*(I-1)/2.0)
    
    correct_ops = ops_recon.eq(ops.type(torch.bool)).float().sum().item()/ (N*I*(45))
    
    # target_arch = design_space.transfer_graph_to_str(adj.cpu().numpy().squeeze(0), ops.cpu().numpy().squeeze(0))
    # pred_arch = design_space.transfer_graph_to_str(adj_recon.cpu().numpy().squeeze(0), ops_recon.cpu().numpy().squeeze(0))
    
    # target_block_list = target_arch.split(')')[:-1]
    # pred_block_list = pred_arch.split(')')[:-1]
    
    # corret_block = 0
    # for pred in pred_block_list:
    #     if pred in target_block_list:
    #         corret_block +=1
            
    return correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj

def get_accuracy(inputs, targets):
    N, I, _ = inputs[0].shape
    ops_recon, adj_recon = inputs[0], inputs[1]
    ops_recon, adj_recon = torch.nn.Sigmoid()(ops_recon), torch.nn.Sigmoid()(adj_recon)
    ops, adj = targets[0], targets[1]
    adj_recon, adj = adj_recon.triu(1), adj.triu(1)
    pred_ops_mask = (torch.sum((ops_recon>0.5).int(), dim=-1) > 0).int()
    tar_ops_mask = (torch.sum(ops, dim=-1) > 0).int()
    correct_ops = torch.mul(ops_recon.argmax(dim=-1), pred_ops_mask).eq(torch.mul(ops.argmax(dim=-1),tar_ops_mask)).float().mean().item()
    #correct_ops = ops_recon.argmax(dim=-1).eq(ops.argmax(dim=-1)).float().mean().item()
    mean_correct_adj = adj_recon[adj.type(torch.bool)].sum().item() / adj.sum()
    mean_false_positive_adj = adj_recon[(~adj.type(torch.bool)).triu(1)].sum().item() / (N*I*(I-1)/2.0-adj.sum())
    threshold = 0.5 # hard threshold
    adj_recon_thre = adj_recon > threshold
    correct_adj = adj_recon_thre.eq(adj.type(torch.bool)).float().triu(1).sum().item()/ (N*I*(I-1)/2.0)
    
    # ops_recon_thre = ops_recon > 0.5
    # correct_ops = ops_recon_thre[ops.type(torch.bool)].sum().item()/ ops.sum().item()
    #print(correct_ops)

    return correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj

def get_accuracy_extend(inputs, targets):
    N, I, _ = inputs[1].shape
    ops_recon, adj_recon = inputs[0], inputs[1]
    ops_recon, adj_recon = torch.nn.Sigmoid()(ops_recon), torch.nn.Sigmoid()(adj_recon)
    ops, adj = targets[0], targets[1]
    adj_recon, adj = adj_recon.triu(1), adj.triu(1)
    
    
    ops_recon_features = ops_recon[:, :, :-3]
    ops_features = ops[:, :, :-3]
    ops_recon_kernels = ops_recon[:, :, -3:]
    ops_kernels = ops[:, :, -3:]
    
    

    #correct_ops = ops_recon.argmax(dim=-1).eq(ops.argmax(dim=-1)).float().mean().item()
    mean_correct_adj = adj_recon[adj.type(torch.bool)].sum().item() / adj.sum()
    if (N*I*(I-1)/2.0-adj.sum()) != 0:
        mean_false_positive_adj = adj_recon[(~adj.type(torch.bool)).triu(1)].sum().item() / (N*I*(I-1)/2.0-adj.sum())
    else:
        mean_false_positive_adj = 0
    threshold = 0.5 # hard threshold
    adj_recon_thre = adj_recon > threshold
    correct_adj = adj_recon_thre.eq(adj.type(torch.bool)).float().triu(1).sum().item()/ (N*I*(I-1)/2.0)
    
    
    pred_ops_mask = (torch.sum((ops_recon_features>0.5).int(), dim=-1) > 0).int()
    tar_ops_mask = (torch.sum(ops_features, dim=-1) > 0).int()
    correct_ops = torch.mul(ops_recon_features.argmax(dim=-1), pred_ops_mask).eq(torch.mul(ops_features.argmax(dim=-1),tar_ops_mask))
    
    pred_kernel_mask = (torch.sum((ops_recon_kernels>0.5).int(), dim=-1) > 0).int()
    tar_kernel_mask = (torch.sum(ops_kernels, dim=-1) > 0).int()
    correct_kernel = torch.mul(ops_recon_kernels.argmax(dim=-1),pred_kernel_mask).eq(torch.mul(ops_kernels.argmax(dim=-1),tar_kernel_mask))
    
    # print(correct_ops)
    # print(correct_kernel)
    correct_ops = (correct_ops & correct_kernel).float().mean().item()
    # ops_recon_thre = ops_recon > 0.5
    # correct_ops = ops_recon_thre[ops.type(torch.bool)].sum().item()/ ops.sum().item()
    #print(correct_ops)

    return correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj


def get_accuracy_with_chunk_size(inputs, targets, N, I, feature_chunks_size=None):
    ops_recon, adj_recon = inputs[0], inputs[1]
    adj_recon = torch.nn.Sigmoid()(adj_recon)
    ops, adj = targets[0], targets[1]
    adj_recon, adj = adj_recon.triu(1), adj.triu(1)
    mean_correct_adj = adj_recon[adj.type(torch.bool)].sum().item() / adj.sum()
    mean_false_positive_adj = adj_recon[(~adj.type(torch.bool)).triu(1)].sum().item() / (N*I*(I-1)/2.0-adj.sum())
    threshold = 0.5 # hard threshold
    adj_recon_thre = adj_recon > threshold
    correct_adj = adj_recon_thre.eq(adj.type(torch.bool)).float().triu(1).sum().item()/ (N*I*(I-1)/2.0)
    
    targets_chunks =  torch.split(ops, split_size_or_sections=feature_chunks_size, dim=-1)
    
    correct_ops = 0
    for pred, tar in zip(ops_recon, targets_chunks):
        pred = torch.nn.Sigmoid()(pred)
        pred_ops_mask = (torch.sum((pred>0.5).int(), dim=-1) > 0).int()
        tar_ops_mask = (torch.sum(tar, dim=-1) > 0).int()
        correct_ops += torch.mul(pred.argmax(dim=-1), pred_ops_mask).eq(torch.mul(tar.argmax(dim=-1),tar_ops_mask)).float().mean().item()
    
    correct_ops = correct_ops/len(feature_chunks_size)
    return correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj

def pick_gpu_lowest_memory():
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.memory_used)/float(gpu.memory_total), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    return bestGPU

def run_clustering(data_for_clustering, cluster_eps=1):
    data_for_clustering = np.array(data_for_clustering)
    db = DBSCAN(eps=cluster_eps, min_samples=1).fit(data_for_clustering.astype('double'))
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    
    Z=db.fit_predict(data_for_clustering.astype('double'))
    return Z, db, n_clusters_
