import os
import sys
import shutil
import pickle
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import model_for_FLOPs

sys.path.append("../../")
from utils.utils import *
from cal_FLOPs import print_model_parm_flops
from torchvision import datasets, transforms
from torch.autograd import Variable
from mobilenet_v1 import MobileNetV1, channel_scale

sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser("MobileNetV1")
parser.add_argument('--max_iters', type=int, default=20)
parser.add_argument('--net_cache', type=str, default='../training/models/checkpoint.pth.tar', help='model to be loaded')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
parser.add_argument('--save_dict_name', type=str, default='save_dict.txt')
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()

# use a model_for_flops to infer the flops of selected model
# you may also calculate the flops by hand
model_for_flops = model_for_FLOPs.MobileNetV1().cuda()
max_FLOPs = 330

# file for save the intermediate searched results
save_dict = {}
if os.path.exists(args.save_dict_name):
    f = open(args.save_dict_name, 'rb')
    save_dict = pickle.load(f)
    f.close()
    print(save_dict, flush=True)

# load training data
traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# data augmentation
crop_scale = 0.08
lighting_param = 0.1
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
    Lighting(lighting_param),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize])

train_dataset = datasets.ImageFolder(
    traindir,
    transform=train_transforms)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

# load validation data
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

# infer the accuracy of a selected pruned net (identidyed with ids)
def infer(model, criterion, ids):

    # calculate flops using model_for_flops
    global model_for_flops
    model_for_flops = model_for_FLOPs.MobileNetV1().cuda()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # modify the batchnorm parameter
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean = torch.zeros_like(m.running_mean)
            m.running_var = torch.ones_like(m.running_var)
            m.momentum = 0.1

    # recalibrate batchnorm for each selected pruned network
    # we only need to run the forward pass and the statistics of batchnorm will be recalculated
    model.train()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            if i >= 100:
                break
            data_time.update(time.time() - end)
            images = images.cuda()
            target = target.cuda()

            logits = model(images, ids.astype(np.int))
            del logits

    # evaluate the corresponding pruned network
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images, ids.astype(np.int))
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'

              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

# prepare ids for testing
def test_candidates_model(model, criterion, candidates, cnt, test_dict):
    for can in candidates:
        print('test {}th model'.format(cnt), flush=True)
        print(list(can[:-1].astype(np.int32)))
        print('FLOPs = {:.2f}M'.format(can[-1]), flush=True)

        t_can = tuple(can[:-1])
        assert t_can not in test_dict.keys()
        print(t_can, flush=True)

        if t_can in save_dict.keys():
            Top1_err = save_dict[t_can]
            print('Already tested Top1_err = {:.2f}'.format(Top1_err))

        else:
            Top1_acc, Top5_acc, loss = infer(model, criterion, can[:-1])
            Top1_err = 100.0 - Top1_acc
            Top5_err = 100.0 - Top5_acc
            print('Top1_err = {:.2f} Top5_err = {:.2f} loss = {:.4f}'.format(Top1_err, Top5_err, loss), flush=True)
            save_dict[t_can] = Top1_err
        cnt += 1
        assert Top1_err >= 0
        can[-1] = Top1_err
        test_dict[t_can] = can[-1]

    return candidates, cnt

# mutation operation in evolution algorithm
def get_mutation(keep_top_k, num_states, mutation_num, m_prob, test_dict, untest_dict):
    print('mutation ......', flush=True)
    res = []
    k = len(keep_top_k)
    iter = 0
    max_iters = 10
    while len(res)<mutation_num and iter<max_iters:
        ids = np.random.choice(k, mutation_num)
        select_seed = np.array([keep_top_k[id] for id in ids])
        is_m = np.random.choice(np.arange(0,2), (mutation_num, num_states+1), p=[1-m_prob, m_prob])
        mu_val = np.random.choice(np.arange(1,len(channel_scale)), (mutation_num, num_states+1))*is_m
        select_list = ((select_seed + mu_val) % len(channel_scale))
        iter += 1
        for can in select_list:
            t_can = tuple(can[:-1])
            flops = print_model_parm_flops(model_for_flops, can[:-1].astype(np.int))
            if t_can in untest_dict.keys() or t_can in test_dict.keys() or flops>max_FLOPs:
                continue
            can[-1] = flops
            res.append(can)
            untest_dict[t_can] = flops
            if len(res)==mutation_num:
                break

    print('mutation_num = {}'.format(len(res)), flush=True)
    return res

# crossover operation in evolution algorithm
def get_crossover(keep_top_k, num_states, crossover_num, test_dict, untest_dict):
    print('crossover ......', flush=True)
    res = []
    k = len(keep_top_k)
    iter = 0
    max_iters = 10 * crossover_num
    while len(res)<crossover_num and iter<max_iters:
        id1, id2 = np.random.choice(k, 2, replace=False)
        p1 = keep_top_k[id1]
        p2 = keep_top_k[id2]
        mask = np.random.randint(low=0, high=2, size=(num_states+1)).astype(np.float32)
        can = p1*mask + p2*(1.0-mask)
        iter += 1
        t_can = tuple(can[:-1])
        flops = print_model_parm_flops(model_for_flops, can[:-1].astype(np.int))
        if t_can in untest_dict.keys() or t_can in test_dict.keys() or flops>max_FLOPs:
            continue
        can[-1] = flops
        res.append(can)
        untest_dict[t_can] = -1
        if len(res)==crossover_num:
            break
    print('crossover_num = {}'.format(len(res)), flush=True)
    return res

# random operation in evolution algorithm
def random_can(num, num_states, test_dict, untest_dict):
    print('random select ........', flush=True)
    candidates = []
    while(len(candidates))<num:
        can = np.random.randint(low=0, high=len(channel_scale), size=(num_states+1)).astype(np.float32)
        t_can = tuple(can[:-1])
        flops = print_model_parm_flops(model_for_flops, can[:-1].astype(np.int))
        if t_can in test_dict.keys() or t_can in untest_dict.keys() or flops>max_FLOPs:
            continue
        can[-1] = flops
        candidates.append(can)
        untest_dict[t_can] = -1
    print('random_num = {}'.format(len(candidates)), flush=True)
    return candidates

# select topk
def select(candidates, keep_top_k, select_num):
    print('select ......', flush=True)
    res = []
    keep_top_k.extend(candidates)
    keep_top_k = sorted(keep_top_k, key=lambda can:can[-1])
    return keep_top_k[:select_num]

def search(model, criterion, num_states):
    global model_for_flops

    cnt = 1
    select_num = 50
    population_num = 50
    mutation_num = 25
    m_prob = 0.1
    crossover_num = 25
    random_num = population_num - mutation_num - crossover_num

    test_dict = {}
    untest_dict = {}
    keep_top_k = []
    keep_top_50 = []
    print('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_iters = {}'.format(population_num, select_num, mutation_num, crossover_num, random_num, args.max_iters))

    # first 50 candidates are generated randomly
    candidates = random_can(population_num, num_states, test_dict, untest_dict)

    start_iter = 0
    filename = './searching_snapshot.pkl'
    if os.path.exists(filename):
        data = pickle.load(open(filename, 'rb'))
        candidates = data['candidates']
        keep_top_k = data['keep_top_k']
        keep_top_50 = data['keep_top_50']
        start_iter = data['iter'] + 1

    for iter in range(start_iter, args.max_iters):
        model_for_flops = model_for_FLOPs.MobileNetV1().cuda()

        candidates, cnt = test_candidates_model(model, criterion, candidates, cnt, test_dict)
        keep_top_50 = select(candidates, keep_top_50, select_num)
        keep_top_k = keep_top_50[0:10]

        print('iter = {} : top {} result'.format(iter, select_num), flush=True)
        for i in range(select_num):
            res = keep_top_50[i]
            print('No.{} {} Top-1 err = {}'.format(i+1, res[:-1], res[-1]))

        untest_dict = {}
        mutation = get_mutation(keep_top_k, num_states, mutation_num, m_prob, test_dict, untest_dict)
        crossover = get_crossover(keep_top_k, num_states, crossover_num, test_dict, untest_dict)
        random_num = population_num - len(mutation) -len(crossover)
        rand = random_can(random_num, num_states, test_dict, untest_dict)

        candidates = []
        candidates.extend(mutation)
        candidates.extend(crossover)
        candidates.extend(rand)

        print('saving tested_dict ........', flush=True)
        f = open(args.save_dict_name, 'wb')
        pickle.dump(save_dict, f)
        f.close()

        snap = {'candidates':candidates, 'keep_top_k':keep_top_k, 'keep_top_50':keep_top_50, 'iter':iter}
        pickle.dump(snap, open(filename, 'wb'))

    print(keep_top_k)
    print('finish!')

def run():
    t = time.time()
    print('net_cache : ', args.net_cache)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = MobileNetV1()
    model = nn.DataParallel(model.cuda())

    if os.path.exists(args.net_cache):
        print('loading checkpoint {} ..........'.format(args.net_cache))
        checkpoint = torch.load(args.net_cache)
        best_top1_acc = checkpoint['best_top1_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("loaded checkpoint {} epoch = {}" .format(args.net_cache, checkpoint['epoch']))

    else:
        print('can not find {} '.format(args.net_cache))
        return

    num_states = 13
    search(model, criterion, num_states)

    total_searching_time = time.time() - t
    print('total searching time = {:.2f} hours'.format(total_searching_time/3600), flush=True)


if __name__ == '__main__':
    run()


