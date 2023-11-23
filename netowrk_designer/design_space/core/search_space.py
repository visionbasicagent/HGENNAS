import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')

import copy 
import random
from tqdm import tqdm
import numpy as np

from abc import abstractmethod
from network_designer.design_space.core.utils import parent_combinations as parent_combinations_old
from network_designer.design_space.core.utils import graph_hash_np

import networkx as nx
from networkx.algorithms.dag import lexicographical_topological_sort

NF_GRAPH_SEARCH_201 = ['skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
NF_GRAPH_SEARCH_101 = ['skip_connect', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']

SearchSpaceNames = {
    'nasbench201': NF_GRAPH_SEARCH_201,
    'nasbench101': NF_GRAPH_SEARCH_101   
}

def expand_with_global_op(adjacency_matrix_size, adj):
    adj = list(adj)
    global_row = [1 for i in range(adjacency_matrix_size)]
    adj.insert(0, np.array(global_row))

    pad_adj = np.zeros([adjacency_matrix_size+1, adjacency_matrix_size+1])

    pad_adj[:, 1:] = np.array(adj)
    # Add diag matrix
    for idx, row in enumerate(pad_adj):
        row[idx] = 1

    return np.array(pad_adj)

def optimised_graph_with_lexicographical_topological_sort(adj):
    row_sum = np.sum(adj, axis=1) > 0
    col_sum = np.sum(adj, axis=0) > 0
    
    leaf_node = list(np.flatnonzero(row_sum != col_sum))

    for n in leaf_node:
        #input and output always leaf node
        if n == 0 or n == adj.shape[-1]-1:
            continue
        else:
            adj[:,n] = adj[:,n]*0
            adj[n,:] = adj[n,:]*0
            
    graph = nx.from_numpy_matrix(np.triu(adj), create_using=nx.DiGraph)
                
    #remove connection with no connection from input
    valid_nodes = [0]
    for node_idx in lexicographical_topological_sort(graph):
        predecessors = [i for i in graph.predecessors(node_idx)]
        if len(predecessors) != 0:
            valid_nodes.append(node_idx)
            for p in predecessors:
                if p not in valid_nodes:
                    adj[p, node_idx] = 0
    
    return adj
    
class SearchSpace:

    def __init__(self, num_intermediate_nodes, num_canidate_ops, sample_skip=True):
        self.num_intermediate_nodes = num_intermediate_nodes
        self.num_parents_per_node = {
            '0':0
        }
        self.adjacency_matrix_size = num_intermediate_nodes+2
        self.minimum_edges = 5
        self.num_canidate_ops = num_canidate_ops
        
        if sample_skip:
            self.ops_start = 0
        else:
            self.ops_start = 1
            
    @abstractmethod
    def create_adjacency_matrix(self, parents, **kwargs):
        """Based on given connectivity pattern create the corresponding adjececny matrix"""
        pass
    
    def _generate_adjacency_matrix(self, adjacency_matrix, node):
        if self._check_validity_of_adjacency_matrix(adjacency_matrix):
            yield adjacency_matrix
        else:
            req_num_parents = self.num_parents_per_node[str(node)]
            current_num_parants = np.sum(adjacency_matrix[:, node], dtype=np.int)
            num_parents_left = req_num_parents - current_num_parants
            
            for parents in parent_combinations_old(adjacency_matrix, node, n_parents=num_parents_left):
                adjacency_matrix_copy = copy.copy(adjacency_matrix)
                for parent in parents:
                    adjacency_matrix_copy[parent, node]=1
                    for graph in self._generate_adjacency_matrix(adjacency_matrix=adjacency_matrix_copy, node=parent):
                        yield graph
    
    def generate_adjacency_matrix_without_loss_ends(self, upscale=False):
        for adjacency_matrix in self._generate_adjacency_matrix(adjacency_matrix=np.zeros([self.adjacency_matrix_size, self.adjacency_matrix_size]), node=(self.adjacency_matrix_size-1)):
            yield adjacency_matrix
            
    def _optimise_graph_with_skip(self, adj, ops):
        adj = list(adj.astype(int))
        for row_id in range(len(adj)-1):
            for i in range(row_id+1, len(list(adj[row_id]))):
                #if node output connect to a skip_connection
                if adj[row_id][i]>0 and ops[i] == 0:
                    adj[row_id][i]=0
                    #and directly connect to its predecessors 
                    for col in range(len(adj[i])):
                        if adj[i][col] >0 :
                            adj[row_id][col] = 1 
        return np.array(adj)
    
    def sample(self):
        #sample num_parents in each node
        for node in range(1,self.adjacency_matrix_size):
            #atleast has one parent make it connected
            self.num_parents_per_node[str(node)] = random.randint(1, node)
        
        #print(self.num_parents_per_node)
        adjacency_matrix_sample = self._sample_adjaceny_matrix_without_loose_ends(
            adjacency_matrix=np.zeros([self.adjacency_matrix_size, self.adjacency_matrix_size]),
            node=self.num_intermediate_nodes+1)
        
        if not self._check_validity_of_adjacency_matrix(adjacency_matrix_sample):
            return self.sample()

        ops = random.choices(list(range(self.ops_start, self.num_canidate_ops)), k=self.num_intermediate_nodes)
        
        ops.insert(0, self.num_canidate_ops)
        ops.append(self.num_canidate_ops+1)
  
        # Create features matrix from labels
        # overall col for ops features will be self.num_cadidate_ops -1 + 3, -1 for remove skip and +3 for global input and output
        # overall row will be num_itermidiate_nodes+global+input+output
        features = [[0 for _ in range(self.num_canidate_ops+2)] for _ in range(self.num_intermediate_nodes+3)]
        features[0][self.num_canidate_ops+1] = 1 # global
        features[1][self.num_canidate_ops-1] = 1 # input
        features[-1][self.num_canidate_ops] = 1 # output
        for idx, op in enumerate(ops):
            if op != 0 and op !=self.num_canidate_ops and op!=(self.num_canidate_ops+1):
                features[idx+1][int(op)-1] = 1
        
        adjacency_matrix_sample = self._optimise_graph_with_skip(adjacency_matrix_sample, ops)
        adjacency_matrix_sample = optimised_graph_with_lexicographical_topological_sort(adjacency_matrix_sample)
        adjacency_matrix_sample = expand_with_global_op(self.adjacency_matrix_size, adjacency_matrix_sample)
        
        return adjacency_matrix_sample, np.array(features)
    
    def _sample_adjaceny_matrix_without_loose_ends(self, adjacency_matrix, node):
        req_num_parents = self.num_parents_per_node[str(node)]
        current_num_parents = np.sum(adjacency_matrix[:, node], dtype=np.int)

        num_parents_left = req_num_parents - current_num_parents
        sampled_parents = random.sample(list(parent_combinations_old(adjacency_matrix, node, n_parents=num_parents_left)), 1)[0]
        for parent in sampled_parents:
            adjacency_matrix[parent, node] = 1
            adjacency_matrix = self._sample_adjaceny_matrix_without_loose_ends(adjacency_matrix, parent)
        return adjacency_matrix
            
    def _check_validity_of_adjacency_matrix(self, adjacency_matrix):
        """
        Checks whether a graph is a valid graph in the search space.
        1. Checks that the graph is non empty
        2. Checks that every node has the correct number of inputs
        3. Checks that if a node has outgoing edges then it should also have incoming edges
        4. Checks that input node is connected
        5. Checks that the graph has no more than self.num_edges edges
        :param adjacency_matrix:
        :return:
        """
        #Check that graph contrains nodes
        num_itermediate_nodes = sum(np.array(np.sum(adjacency_matrix, axis=1) > 0, dtype=int)[1:-1])
        if num_itermediate_nodes == 0:
            return False
        
        #Check that every node has extactly the right number of inputs
        col_sums = np.sum(adjacency_matrix[:, :], axis=0)
        for col_idx, col_sum in enumerate(col_sums):
            if col_sum > 0:
                if col_sum != self.num_parents_per_node[(str(col_idx))]:
                    return False
                
        #Check that if a node has outputs then it should also have incoming edges
        col_sums = np.sum(np.sum(adjacency_matrix, axis=0) > 0)
        row_sums = np.sum(np.sum(adjacency_matrix, axis=1) > 0)
        if col_sums != row_sums:
            return False
        
        #Check that the input node is always connected. Otherwise the graph is disconnected.
        row_sum = np.sum(adjacency_matrix, axis=1)
        if row_sum[0] == 0:
            return False
        
        #Check that the graph returned has more than self.num_edges_limits: edges.
        num_edges = np.sum(adjacency_matrix.flatten())
        if num_edges < self.minimum_edges:
            return False
        
        return True

    def check_validity_of_adjacency_matrix(self, adjacency_matrix):
        """
        Checks whether a graph is a valid graph in the search space.
        1. Checks that the graph is non empty
        3. Checks that if a node has outgoing edges then it should also have incoming edges
        4. Checks that input node is connected
        5. Checks that the graph has no more than self.num_edges edges
        :param adjacency_matrix:
        :return:
        """
        #print(adjacency_matrix)
        #Check that graph contrains nodes
        num_itermediate_nodes = sum(np.array(np.sum(adjacency_matrix, axis=1) > 0, dtype=int)[1:-1])
        if num_itermediate_nodes == 0:
            return False
        
        # #Check that every node has at least 1 inputs, except input node
        # col_sums = np.sum(adjacency_matrix[:, :], axis=0)
        # for col_idx, col_sum in enumerate(col_sums):
        #     if col_idx > 0:
        #         if col_sum == 0:
        #             return False

        # #Check that if a node has outputs then it should also have incoming edges
        # col_sums = np.sum(np.sum(adjacency_matrix, axis=0) > 0)
        # row_sums = np.sum(np.sum(adjacency_matrix, axis=1) > 0)
        # if col_sums != row_sums:
        #     return False
        
        #Check that the input node is always connected. Otherwise the graph is disconnected.
        row_sum = np.sum(adjacency_matrix, axis=1)
        if row_sum[0] == 0:
            return False
        
        #Check that the output node is always connected.Otherwise the graph is disconnected.
        col_sums = np.sum(adjacency_matrix, axis=0)
        if col_sums[-1] == 0:
            return False
    
        #Check that the graph returned has no more than self.num_edges_limits: edges.
        num_edges = np.sum(adjacency_matrix.flatten())
        #print(num_edges)
        if num_edges < self.minimum_edges:
            return False
        
        return True
        
def sample_graphs(sample_size, notebook=False, seed=None, num_intermediate_node=6, num_candidate_ops=4, sample_skip=True, search_space=None):
    random.seed(seed)
    ss = SearchSpace(num_intermediate_node, num_candidate_ops, sample_skip=sample_skip)  

    enc = []
    adj_matrix = []
    ops_features = []
    hash_list = []
    # if notebook:
    #     from tqdm.notebook import tqdm
    for i in tqdm(range(sample_size)):
        adj, ops = ss.sample()
        ops = ops.astype(int)
        adj = adj.astype(int)
        hash = calc_graph_hash(adj, ops)
        if hash not in hash_list:
            hash_list.append(hash)

            ops_features.append(ops)
            adj_matrix.append(adj)

    return adj_matrix, ops_features

def calc_graph_hash(adj_matrix, ops_features):
    adj_matrix = np.array(adj_matrix)
    ops_features = list(ops_features)     
    ops = [np.argmax(x) if np.sum(x)!=0 else -1 for x in ops_features]
    #print(adj_matrix)
    #print(ops)
    return graph_hash_np(adjacency_matrix=adj_matrix, ops=ops)

def mutate_arch(adj_matrix, ops_features):
    mutation_type = np.random.choice(['hidden_state_mutation', 'op_mutation'])
    
    adj_matrix_copy = copy.deepcopy(adj_matrix)
    ops_features_copy = copy.deepcopy(ops_features)
    
    if mutation_type == 'hidden_state_mutation':
        adj_matrix_copy = adj_matrix_copy.astype(int)
        adj_matrix_copy = np.triu(adj_matrix_copy, 1)
        adj_matrix_copy = adj_matrix_copy[1:,1:]
        #do not mutate input node
        low = 1
        random_node = np.random.randint(low=low, high=adj_matrix_copy.shape[-1])
        
        parent_of_node = adj_matrix_copy[:random_node, random_node].nonzero()[0]
        
        if len(parent_of_node) > 0:
            parent_of_node_to_modify = np.random.choice(parent_of_node)
            
        #find a parent with input 
        candidate_parent_list = list(np.sum(adj_matrix_copy, axis=0)[:random_node].nonzero()[0])
        #you can always pick 0 as parent 
        candidate_parent_list.append(0)
        #remove existing parent
        if len(parent_of_node) > 0:
            for p in parent_of_node:
                if p in candidate_parent_list:
                    candidate_parent_list.remove(p)
        
        new_parent_of_node = None
        if len(candidate_parent_list) > 0:
            new_parent_of_node =np.random.choice(candidate_parent_list)
        
        #remove if has parent
        if len(parent_of_node) > 0 and np.random.choice([True, False]):
            adj_matrix_copy[parent_of_node_to_modify, random_node] = 0
        
        if np.random.choice([True, False]) and (new_parent_of_node is not None):
            #add new parent
            print('add new parent {} to current node {}'.format(new_parent_of_node, random_node))
            adj_matrix_copy[new_parent_of_node, random_node] = 1

            #check if node has output 
            col_sum = np.sum(adj_matrix_copy, axis=1) 

            if col_sum[random_node] == 0:
                #if not has output random pick one with output or directly connected to output
                if len(col_sum[random_node:].nonzero()[0]) == 0:
                    child_node = adj_matrix_copy.shape[-1]-1
                else:
                    child_node = np.random.choice(col_sum[random_node:].nonzero()[0]) + random_node
                adj_matrix_copy[random_node, child_node] = 1
        
        adj_matrix_copy = expand_with_global_op(adj_matrix_copy.shape[-1], adj_matrix_copy)
            
    else:
        ops_features_copy = ops_features_copy.astype(int)
        ops_features_copy = list(ops_features_copy)
        #ignore global, input and output
        random_op_idx = np.random.randint(2, len(ops_features_copy)-1)
        
        original_op = np.argmax(ops_features_copy[random_op_idx][:-3])
        new_op = np.random.randint(len(list(ops_features_copy[random_op_idx][:-3]))-3)
        print('replace op {} from {} to {}'.format(random_op_idx, original_op, new_op))

        
        original_kernel = np.argmax(ops_features_copy[random_op_idx][-3:])
        new_kernel = np.random.randint(len(list(ops_features_copy[random_op_idx][-3:])))
        print('replace op {} from kernel id {} to kernel id {}'.format(random_op_idx, original_kernel, new_kernel))
        
        ops_features_copy = np.array(ops_features_copy)
        ops_features_copy[random_op_idx][:-3][original_op]=0
        ops_features_copy[random_op_idx][:-3][new_op]=1  
        
        ops_features_copy[random_op_idx][-3:][original_kernel]=0
        ops_features_copy[random_op_idx][-3:][new_kernel]=1
              
    return adj_matrix_copy, ops_features_copy
