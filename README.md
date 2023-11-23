# HGENNAS: Towards Neural Architecture Search through Hierarchical Generative Modeling
## Official Implementation for ICLR Submission 6512

### Abstract
Neural Architecture Search (NAS) is gaining popularity in automating designing deep neural networks for various tasks. A typical NAS pipeline begins with a manually designed search space which is methodically explored during the process, aiding the discovery of high-performance models.
Although NAS has shown impressive results in many cases, the strong performance remains largely dependent on, among other things, the prior knowledge about good designs which is implicitly incorporated into the process by carefully designing search spaces. In general, this dependency is undesired, as it limits the applicability of NAS to less-studied tasks and/or results in an explosion of the cost needed to obtain strong results.
In this work, our aim is to address this limitation by leaning on the recent advances in generative modelling -- we propose a method that can navigate an extremely large, general-purpose search space efficiently, by training a two-level hierarchy of generative models. The first level focuses on micro-cell design and leverages Conditional Continuous Normalizing Flow (CCNF) and the subsequent level uses a transformer-based sequence generator to produce macro architectures for a given task and architectural constraints.
To make the process computationally feasible, we perform task-agnostic pretraining of the generative models using a metric space of graphs and their zero-cost (ZC) similarity. We evaluate our method on typical tasks, including CIFAR-10, CIFAR-100 and ImageNet models, where we show state-of-the-art performance compared to other low-cost NAS approaches.

### Overview
The full training scripts are not fully done with clean-up, we now provide the training scritps to re-produce the network we searched for CIFAR and ImageNet-1k tasks 


### System Requirements
PyTorch >= 1.8, Python >= 3.6 

```bash
conda env create -f environment.yml
conda activate gnet
HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]
```

### Reproduce Our Results

```bash
cd network_designer/trainner/sota_trainner/sota_trainner/
##CIFAR Results
train_cifar.sh
##ImageNet-1k Flops<450M> takes 8 GPU to run, we trained with 8x A100 
train_im_450.sh

##ImageNet-1k Flops<600M> takes 8 GPU to run, we trained with 8x A100 
train_im_600.sh

```
### Open-Source
In the development of HGENNAS, we referred to several open-source repositories. These projects have contributed either directly or indirectly to our understanding and implementation. We highly recommend checking out these repositories for further insights:

```
https://github.com/idstcv/ZenNAS
https://github.com/SLDGroup/ZiCo
https://github.com/minimaxir/aitextgen
```
