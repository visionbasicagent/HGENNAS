import os
import sys
import random
import utils as train_utils
sys.path.insert(0, '../../')
sys.path.insert(0, '../../../')
import glob
import numpy as np
import torch

import logging
import argparse
import shutil
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import pickle
import pandas as pd
from network_designer.design_space.extend.search_space import SearchSpaceNames
from network_designer.design_space.extend.model import Network

from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../../data',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='auto', help='gpu device id')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--exp_path', type=str, default='../../experiments/nb201-like-train/', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--space_root', type=str, default='../experiments/DEMO/step_2/')
parser.add_argument('--space', type=str, default='', help="pickle file name for sampled space")
parser.add_argument('--id', type=int, default=0, help="id for architecutres in space")
parser.add_argument('--seed', type=int, default=888, help='random seed')
parser.add_argument('--search_ops_space', type=str, default='extend', help='ops in search space')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
#### common
parser.add_argument('--resume_epoch', type=int, default=0, help="load ckpt, start training at resume_epoch")
parser.add_argument('--ckpt_interval', type=int, default=50, help="interval (epoch) for saving checkpoints")
parser.add_argument('--resume_expid', type=str, default='', help="full expid to resume from, name == ckpt folder name")
parser.add_argument('--fast', action='store_true', default=False, help="fast mode for debugging")

args = parser.parse_args()

#### args augment
expid = args.save

def load_graph_from_pickle(space_root, space, id):
    f = open('{}/{}.pkl'.format(space_root, space), 'rb')
    print(f)
    data = pickle.load(f)
    dataset = pd.DataFrame(data)

    if space.startswith('ea_graph_with_vec'):
        dataset = dataset.sort_values(by='zc_score', ascending=False, na_position='last')
    
    print(dataset.iloc[id])
    adj_matrix = dataset.iloc[id]['adj_matrix']
    adj_matrix = np.triu(adj_matrix, 1)
    ops = dataset.iloc[id]['ops_features']

    if space == 'ref_best' or space == 'pareto' or space == 'pareto_all' or space == 'pareto_latency' or space == 'ea_extend_sort' or  space.startswith('ea_graph_with_vec') or space.startswith('zc_front_consent'):
        adj_matrix = np.triu(adj_matrix, 1)
        adj_matrix = adj_matrix[1:,1:]
        ops = ops[1:]

    if space == '045anchor' or space == '070anchor' or space =='110anchor':
        adj_matrix = np.triu(adj_matrix, 1)
        adj_matrix = adj_matrix[1:,1:]
        ops = ops[1:]

    return adj_matrix, ops
    
args.save = '{}/{}/{}/{}-{}'.format(args.exp_path, args.dataset, args.space, args.seed, args.id)

if args.cutout:
    args.save += '-cutout-' + str(args.cutout_length) + '-' + str(args.cutout_prob)

#### logging
if args.resume_epoch > 0: # do not delete dir if resume:
    args.save = '{}/{}/{}'.format(args.exp_path, args.dataset, args.resume_expid)
    assert(os.path.exists(args.save), 'resume but {} does not exist!'.format(args.save))
else:
    scripts_to_save = glob.glob('*.py')
    if os.path.exists(args.save):
        if input("WARNING: {} exists, override?[y/n]".format(args.save)) == 'y':
            print('proceed to override saving directory')
            shutil.rmtree(args.save)
        else:
            exit(0)
    train_utils.create_exp_dir(args.save, scripts_to_save=scripts_to_save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
log_file = 'log_resume_{}.txt'.format(args.resume_epoch) if args.resume_epoch > 0 else 'log.txt'
fh = logging.FileHandler(os.path.join(args.save, log_file), mode='w')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save + '/runs')


if args.dataset == 'cifar100':
    n_classes = 100
else:
    n_classes = 10

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def main():
    
    adj_matrix, ops = load_graph_from_pickle(args.space_root, args.space, args.id)
    
    # print(adj_matrix)
    # print(ops)
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
 
    gpu = train_utils.pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
    #gpu = 0
    torch.cuda.set_device(gpu)
    cudnn.enabled = True
    seed_torch(args.seed)

    logging.info('gpu device = %d' % gpu)
    logging.info("args = %s", args)

    logging.info(adj_matrix)
    logging.info(ops)

    search_space = SearchSpaceNames[args.search_ops_space]
    model = Network(C=args.init_channels, N=5, num_classes=n_classes, search_space=search_space, adj_matrix=adj_matrix, ops=ops)
    
    model = model.cuda()
    logging.info(model)
    
    logging.info("param size = %fMB", train_utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    if args.dataset == 'cifar10':
        train_transform, valid_transform = train_utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = train_utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'svhn':
        train_transform, valid_transform = train_utils._data_transforms_svhn(args)
        train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
        valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)
    elif args.dataset == 'imagenet16-120':
        import torchvision.transforms as transforms
        from .DownsampledImageNet import ImageNet16
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22,  61.26, 65.09]]
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
        train_transform = transforms.Compose(lists)
        train_data = ImageNet16(root=os.path.join(args.data,'imagenet16'), train=True, transform=train_transform, use_num_of_class_only=120)
        valid_data = ImageNet16(root=os.path.join(args.data,'imagenet16'), train=False, transform=train_transform, use_num_of_class_only=120)
        assert len(train_data) == 151700

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs)*len(train_queue))


    #### resume
    start_epoch = 0
    if args.resume_epoch > 0:
        logging.info('loading checkpoint from {}'.format(expid))
        filename = os.path.join(args.save, 'checkpoint_{}.pth.tar'.format(args.resume_epoch))

        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')
            resume_epoch = checkpoint['epoch'] # epoch
            model.load_state_dict(checkpoint['state_dict']) # model
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer']) # optimizer
            start_epoch = args.resume_epoch
            print("=> loaded checkpoint '{}' (epoch {})".format(filename, resume_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(filename))


    #### main training
    best_valid_acc = 0
    for epoch in range(start_epoch, args.epochs):
        lr = scheduler.get_lr()[0]
        if args.cutout:
            train_transform.transforms[-1].cutout_prob = args.cutout_prob
            logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                         train_transform.transforms[-1].cutout_prob)
        else:
            logging.info('epoch %d lr %e', epoch, lr)
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer, scheduler)
        logging.info('train_acc %f', train_acc)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Obj/train', train_obj, epoch)


        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        writer.add_scalar('Acc/valid', valid_acc, epoch)
        writer.add_scalar('Obj/valid', valid_obj, epoch)

        ## checkpoint
        if (epoch + 1) % args.ckpt_interval == 0:
            save_state_dict = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            train_utils.save_checkpoint(save_state_dict, False, args.save, per_epoch=True)

        best_valid_acc = max(best_valid_acc, valid_acc)
    logging.info('best valid_acc %f', best_valid_acc)
    writer.close()


def train(train_queue, model, criterion, optimizer, scheduler):
    objs = train_utils.AvgrageMeter()
    top1 = train_utils.AvgrageMeter()
    top5 = train_utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        #_, logits = model(input)
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        ## scheduler
        scheduler.step()

        prec1, prec5 = train_utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        if args.fast:
            logging.info('//// WARNING: FAST MODE')
            break

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = train_utils.AvgrageMeter()
    top1 = train_utils.AvgrageMeter()
    top5 = train_utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            #_, logits = model(input)
            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = train_utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            
            if args.fast:
                logging.info('//// WARNING: FAST MODE')
                break

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()