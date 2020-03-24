#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""

from __future__ import print_function
import os
import shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from models.TransformerBaseline_NLA2_S import DGCNN_Transformer
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics

import time


#from visualizer.show3d_balls import showpoints


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    elif args.model == 'TransformerBaseline':
        model = DGCNN_Transformer(args).to(device)
    elif args.model == 'TemporalTransformer':
        model = DGCNN_TemporalTransformer(args).to(device)
    elif args.model == 'TemporalTransformer_v2':
        model = DGCNN_TemporalTransformer_v2(args).to(device)
    elif args.model == 'TemporalTransformer_v3':
        model = DGCNN_TemporalTransformer_v3(args).to(device)
    elif args.model == 'pi':
        model = pi_DGCNN(args).to(device)
    elif args.model == 'pi2':
        model = pi_DGCNN_v2(args).to(device)
    elif args.model == 'pipoint':
        model = pipoint_DGCNN(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    #model = nn.DataParallel(model, device_ids=list(range(3)))
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    # get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.use_sgd:
        print("Use SGD")
        optimizer = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss

    best_test_acc = 0

    if args.model_path:
        if os.path.isfile(args.model_path):
            print("=> loading checkpoint '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            print(checkpoint)

            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            #best_prec5 = checkpoint['best_prec5']
    
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint, strict=False)


            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.model_path, args.start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    end = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        

        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []

        batch_time = AverageMeter()
        data_time = AverageMeter()

        end = time.time()
        for i, (data, label) in enumerate(train_loader):
            data_time.update(time.time()-end) 

            data, label = data.to(device), label.to(device).squeeze()

            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            optimizer.zero_grad()

            if args.model in ["pi", "pipoint", "pi2"]:
                logits, atts = model(data)
            elif args.model in ["TransformerBaseline", "TemporalTransformer", "TemporalTransformer_v2", "TemporalTransformer_v3"]:
                logits = model(data)
            else:
                logits, degree = model(data)

            '''
            if args.visualize == True:

                print(args.visualize)

                import matplotlib.pyplot as plt
                #cmap = plt.cm.get_cmap("hsv", 30)
                cmap = plt.cm.get_cmap("binary", 40)
                cmap = np.array([cmap(i) for i in range(40)])[:,:3]
                obj = degree[7,:,:3].cpu().numpy()
                obj_degree = degree[7,:,3:].squeeze()
                obj_degree = obj_degree.cpu().numpy().astype(int)
              
                obj_max = np.max(obj_degree)
                obj_min = np.min(obj_degree)

                print(obj_max)
                print(obj_min)

                for i in range(obj_min, obj_max):
                    print("{} : {}".format(i, sum(obj_degree == i)))


                gt = cmap[obj_degree-obj_min, :]
                showpoints(obj, gt, gt, ballradius=3)
            '''
            loss = criterion(logits, label, smoothing=True)
            loss.backward()
            optimizer.step()

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

            batch_time.update(time.time()-end)
            end=time.time()

            if i % 10 == 0:
                print_str = 'Train {}, loss {}, Time {batch_time.val:.3f} ({batch_time.avg:.3f}), Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(epoch, train_loss*1.0/count, batch_time=batch_time, data_time=data_time)
                print(print_str)


        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)


        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))


        #outstr = 'Train {}, Time {batch_time.val:.3f} ({batch_time.avg:.3f}), Data {data_time.val:.3f} ({data_time.avg:.3f}), loss: {}, train acc: {}, train avg acc: {}'.format(epoch, batch_time=batch_time, data_time=data_time, train_loss*1.0/count, metrics.accuracy_score(train_true, train_pred), metrics.balanced_accuracy_score(train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        with torch.no_grad():
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_pred = []
            test_true = []

            batch_time = AverageMeter()
            losses = AverageMeter()

            end = time.time()
            for j, (data, label) in enumerate(test_loader):
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]

                if args.model in ["pi", "pipoint", "pi2"]:
                    logits, atts = model(data)
                elif args.model in ["TransformerBaseline", "TemporalTransformer", "TemporalTransformer_v2", "TemporalTransformer_v3"]:
                    logits = model(data)
                else:
                    logits, degree = model(data)


                loss = criterion(logits, label, smoothing=True)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

                batch_time.update(time.time() - end)
                end = time.time()

                if j % 10 == 0:
                    print('Test {}, Loss {}, Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(j, test_loss*1.0/count, batch_time=batch_time))


            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

            #per_class_acc = metrics.precision_score(test_true, test_pred, average=None)
            #outstr = 'Test {}, loss: {}, train acc: {}, train avg acc: {}'.format(epoch, batch_time=batch_time, data_time=data_time, test_loss*1.0/count, test_acc, avg_per_class_acc)
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
            #outstr_2 = 'Test per class acc: {}'%per_class_acc
            io.cprint(outstr)
            #io.cprint(outstr_2)
            if args.model in ["pi", "pipoint", "pi2"]:
                for j in range(4):
                    io.cprint('Att {} : {}'.format(j, atts[j].mean().item()))

            is_best = test_acc >= best_test_acc

            if is_best:
                best_test_acc = test_acc

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, is_best, args.exp_name)

            #torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = torch.softmax(output, dim = 1)
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def save_checkpoint(state, is_best, prefix):
    filename='./checkpoints/%s/models/model.tar'%args.exp_name
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/%s/models/model_best.tar'%args.exp_name)



def test(args, io):
    #test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
    #                         batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    elif args.model == 'TransformerBaseline':
        model = DGCNN_Transformer(args).to(device)
    elif args.model == 'TemporalTransformer':
        model = DGCNN_TemporalTransformer(args).to(device)
    elif args.model == 'TemporalTransformer_v2':
        model = DGCNN_TemporalTransformer_v2(args).to(device)
    elif args.model == 'TemporalTransformer_v3':
        model = DGCNN_TemporalTransformer_v3(args).to(device)
    elif args.model == 'pi':
        model = pi_DGCNN(args).to(device)
    elif args.model == 'pi2':
        model = pi_DGCNN_v2(args).to(device)
    elif args.model == 'pipoint':
        model = pipoint_DGCNN(args).to(device)
    else:
        raise Exception("Not implemented")

    print(model)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    model = nn.DataParallel(model)

    if args.model_path:
        if os.path.isfile(args.model_path):
            print("=> loading checkpoint '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            print(checkpoint)

            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            #best_prec5 = checkpoint['best_prec5']
    
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.model_path, args.start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.model_path))
    #model.load_state_dict(torch.load(args.model_path))

    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []

    batch_time = AverageMeter()

    
    end = time.time()
    for i, (data, label) in enumerate(test_loader):

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]

        if args.model in ["pi", "pipoint", "pi2"]:
            logits, atts = model(data)
        elif args.model in ["TransformerBaseline", "TemporalTransformer", "TemporalTransformer_v2"]:
            logits = model(data)
        else:
            logits, degree = model(data)

        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test {}, Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(i, batch_time=batch_time))

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

    per_class_acc = metrics.precision_score(test_true, test_pred, average=None)


    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    outstr_2 = 'Test per class acc: {}'%per_class_acc
    io.cprint(outstr)
    io.cprint(outstr_2)

    if args.model in ["pi", "pipoint", "pi2"]:
        for j in range(4):
            io.cprint('Att {} : {}'.format(j, atts[j].mean().item()))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn', 'pi', 'pi2', 'pipoint', 'TransformerBaseline', 'TemporalTransformer', 'TemporalTransformer_v2', 'TemporalTransformer_v3'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--visualize', type=bool, default=False,
                        help='Pretrained model path')
    parser.add_argument('--deg', action='store_true',
                        help='Pretrained model path')
    parser.add_argument('--delta', action='store_true',
                        help='Pretrained model path')
    parser.add_argument('--neighbors', action='store_true',
                        help='Pretrained model path')
    parser.add_argument('--rpe', action='store_true',
                        help='Pretrained model path')
    parser.add_argument('--ape', action='store_true',
                        help='Pretrained model path')
    parser.add_argument('--scale', action='store_true',
                        help='Pretrained model path')
    parser.add_argument('--dimkqv', type=int,
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
