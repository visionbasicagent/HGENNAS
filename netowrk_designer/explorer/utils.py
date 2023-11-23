import os
import logging
import pandas as pd
import numpy as np

def cache_space_record(history, path, save_index, mutate_type, budget_model_size=None, budget_flops=None, dataset='cifar10', post_fix='child', folder='masterspace'):
    valid_adjs = []
    valid_opss = []
    params_list = []
    flops_list = []
    zc_score_1 = []
    zc_score_2 = []
    layer_channels_list = []
    layer_stride_list = []
    layer_repeats_list = []
    iter_index = save_index
    save_iters = []
    final_concat_layer_list = []
    
    for his in history:
        for sampled_model in his.population:
            params_list.append(sampled_model.params)
            flops_list.append(sampled_model.flops)
            valid_adjs.append(sampled_model.adjs)
            valid_opss.append(sampled_model.opss)
            zc_score_1.append(sampled_model.score[0])
            #zc_score_2.append(sampled_model.score[1]) 
            layer_channels_list.append(sampled_model.layer_channels)
            layer_stride_list.append(sampled_model.layer_stride)
            layer_repeats_list.append(sampled_model.layer_repeats)
            final_concat_layer_list.append(sampled_model.final_concat_layer)
            save_iters.append(iter_index)
        iter_index += 1

        
        
    df = pd.DataFrame(
    {
        'adj_matrix': valid_adjs,
        'ops_features': valid_opss,
        'layer_channels': layer_channels_list,
        'layer_stride': layer_stride_list,
        'layer_repeats': layer_repeats_list,
        'save_iter': save_iters,
        'final_concat_layer':final_concat_layer_list
    })
    
    df['zc_score_1'] = zc_score_1
    #df['zc_score_2'] = zc_score_2
    df['param'] = params_list 
    df['flops'] = flops_list

    print(df)

    save_path = os.path.join(path, folder)
    os.makedirs(save_path, exist_ok=True)
    
    if budget_model_size:
        df.to_pickle(os.path.join(save_path, 'sapce_exploration_params_{}_ckpt_{}_{}_{}.pkl'.format(budget_model_size, save_index, post_fix,dataset)))  
        
    if budget_flops:
        df.to_pickle(os.path.join(save_path, 'sapce_exploration_flops_{}_ckpt_{}_{}_{}.pkl'.format(budget_flops, save_index, post_fix ,dataset)))
        

def cache_arch_record(history, path, save_index, mutate_type, budget_model_size=None, budget_flops=None, dataset='cifar10', post_fix='child', folder='masternet'):
    valid_adjs = []
    valid_opss = []
    params_list = []
    flops_list = []
    zc_score_1 = []
    zc_score_2 = []
    layer_channels_list = []
    layer_stride_list = []
    layer_repeats_list = []
    final_concat_layer_list=[]
    
    for his in history:
        params_list.append(his.params)
        flops_list.append(his.flops)
        valid_adjs.append(his.adjs)
        valid_opss.append(his.opss)
        zc_score_1.append(his.score[0])
        #zc_score_2.append(his.score[1]) 
        layer_channels_list.append(his.layer_channels)
        layer_stride_list.append(his.layer_stride)
        layer_repeats_list.append(his.layer_repeats)
        final_concat_layer_list.append(his.final_concat_layer)

        
        
    df = pd.DataFrame(
    {
        'adj_matrix': valid_adjs,
        'ops_features': valid_opss,
        'layer_channels': layer_channels_list,
        'layer_stride': layer_stride_list,
        'layer_repeats': layer_repeats_list,
        'final_concat_layer':final_concat_layer_list,
    })
    
    df['zc_score_1'] = zc_score_1
    #df['zc_score_2'] = zc_score_2
    df['param'] = params_list 
    df['flops'] = flops_list

    save_path = os.path.join(path, folder)
    os.makedirs(save_path, exist_ok=True)
    
    if budget_model_size:
        df.to_pickle(os.path.join(save_path, 'space_explitation_params_{}_ckpt_{}_{}_{}.pkl'.format(budget_model_size, save_index, post_fix,dataset)))  
        
    if budget_flops:
        df.to_pickle(os.path.join(save_path, 'space_explitation_flops_{}_ckpt_{}_{}_{}.pkl'.format(budget_flops, save_index, post_fix,dataset)))

def create_logging(log_filename=None, level=logging.INFO):
    if log_filename is not None:
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[
                logging.StreamHandler()
            ]
        )

def count_parameters(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)

def pick_gpu_lowest_memory():
    import gpustat
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.memory_used)/float(gpu.memory_total), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    return bestGPU