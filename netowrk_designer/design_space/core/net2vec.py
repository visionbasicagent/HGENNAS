try:
    from foresight.pruners import predictive
except:
    pass
import torch
import itertools
import numpy as np
try:
    from alethiometer import calc_zc_metrics  
except:
    pass
def get_synflow(network, xloader, num_batch=1, classes=10, type='synflow_SSNRs'):
    measures = predictive.find_measures(network,
                                        xloader,
                                        ('random', 1, classes), 
                                        torch.cuda.current_device(),
                                        measure_names=['synflow'], aggregate=False
                )
    measures = measures['synflow']
    #print(measures)
    scores = {}
    for k, v in measures.items():
        s = v.detach().view(-1) 
        if torch.std(s) == 0:
            s = torch.sum(s)
        else:
            s = torch.sum(s)/torch.std(s)
        scores[k]=s.cpu().numpy()
        
    return scores   

def summarize_micro_node_vector(synflow_dict, node_info):
    zc_vec = []
    global_var = 0
    for k, v in synflow_dict.items():
        global_var += v
        
    for node, op_names in node_info.items():
        vec = []
        for stage in op_names:
            stage_var = 0
            for layer in stage:
                #print(layer)
                if layer in synflow_dict:
                    stage_var += synflow_dict[layer]    
            vec.append(stage_var)
        zc_vec.append(sum(vec))
    #print(zc_vec)
    global_var = global_var - np.sum(np.array(zc_vec))
    zc_vec.append(global_var)
    return np.array(zc_vec)

def summarize_macro_node_vector(synflow_dict, node_info):
    zc_vec = []
    for node, op_names in node_info.items():
        node_var = 0
        for op in op_names:
            if op in synflow_dict:
                node_var += synflow_dict[op]
        zc_vec.append(node_var)
    return np.array(zc_vec)

def summarize_block_vector(synflow_dict, block_info):
    raw_encoding = list(itertools.repeat(0, len(block_info)))
    for i, pos in enumerate(block_info):
        for n, m in synflow_dict.items():
            if n.startswith(pos):
                if m != 0:
                    raw_encoding[i] += m
    # for i in range(len(raw_encoding)):
    #     raw_encoding[i] = np.log1p(raw_encoding[i])
    return np.array(raw_encoding)

def summarize_flat_vector(synflow_dict, flat_info, max_len=64):
    assert len(flat_info) < max_len, 'layers numbers:{} great than max_len:{}'.format(len(flat_info), max_len)
    zc_vec = np.zeros(max_len)
    for i, n in enumerate(flat_info):
        if n in synflow_dict:
            zc_vec[i] = synflow_dict[n]
    return zc_vec

def get_zc_vec(net, xloader, device=torch.cuda.current_device(), micro=True):
    if micro:
        zc_score_1 = 'snr_snip_none'
        zc_score_2 = 'nwot'
        
        results = calc_zc_metrics(metrics=[zc_score_1, zc_score_2], model=net, train_queue=xloader, device=device, aggregate=True)
        return [results[zc_score_1], results[zc_score_2]]
    else:
        zc_score_1 = 'tcet_snip_log1p'
        
        print(zc_score_1)
            
        #zc_score_2 = 'nwot'
        results = calc_zc_metrics(metrics=[zc_score_1], model=net, train_queue=xloader, device=device, aggregate=True)
        return [results[zc_score_1]]