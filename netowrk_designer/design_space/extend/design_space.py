import random
import sys
from network_designer.design_space.core.design_space_base import DesignSpace
from network_designer.design_space.extend.operations import SearchSpaceNames
from network_designer.design_space.extend.search_space import SearchSpace, calc_graph_hash, mutate_arch

class ExtendDesignSpace(DesignSpace):
    def __init__(self, args, n_classes=10):
        super(ExtendDesignSpace, self).__init__(args=args)
        #network settings
        self.init_channels = 16
        self.N_nodes = 5
        #design space settings
        self.candidate_operations = SearchSpaceNames['nf_graph']
        print(self.candidate_operations)
        self.design_space = SearchSpace(6, 10)

        self.n_classes = n_classes
            
        self.INPUT_NODE = 0
        self.OUTPUT_NODE = 7
        
    def sample_unique_graphs(self, sample_size, seed, notebook=False):
        random.seed(seed)
        adj_matrix = []
        ops_features = []
        hash_list = []
        try_out = 0
        time_out = sample_size * 10
        while(len(hash_list)<sample_size and try_out<time_out):
            adj, ops = self.design_space.sample()
            ops = ops.astype(int)
            adj = adj.astype(int)
            hash = calc_graph_hash(adj, ops)
            if hash not in hash_list:
                hash_list.append(hash)
                ops_features.append(ops)
                adj_matrix.append(adj)
                sys.stdout.write('\r')
                sys.stdout.write('| Sampling [%3d/%3d]' 
                                % (len(hash_list), sample_size))
                sys.stdout.flush()
            try_out+=1

        return adj_matrix, ops_features, hash_list
    
    def sample(self):
        return self.design_space.sample()
        
    def mutate_arch(self, adj, ops):
        return mutate_arch(adj, ops)
            

    
    