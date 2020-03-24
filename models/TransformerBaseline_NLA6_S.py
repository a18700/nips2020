#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import pdb

from .attention_NLA6_S import AttentionConv


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)

    return idx

def get_neighbors(x, k=20, idx=None, delta=True, neighbors=True, static=False, layer=0):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx == None:
        if static is False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
            idx_return = idx.unsqueeze(1)
        else:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
            idx_initial = idx

    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points # 1, 1, 1

    idx = idx + idx_base # B, N, K

    _, num_dims, _ = x.size() # 3
    x = x.transpose(2, 1).contiguous() # B, N, C

    abs_x = x.permute(0,2,1).unsqueeze(3)

    idx = idx.view(-1)

    feature = x.view(batch_size*num_points, -1)[idx, :] # 1024*20, 3
    feature = feature.view(batch_size, num_points, k, -1).permute(0, 3, 1, 2) # 1, 3, 1024, 20

    if delta and neighbors:
        x = x.view(batch_size, num_points, 1, -1).repeat(1, 1, k, 1).permute(0, 3, 1, 2) # 1, 1024, 20, 3
        feature = torch.cat((feature-x, feature), dim=1) # 1, 6, 1024, 20

    elif delta and not neighbors:
        x = x.view(batch_size, num_points, 1, -1).repeat(1, 1, k, 1).permute(0, 3, 1, 2) # 1, 1024, 20, 3
        feature = torch.cat((feature-x, x), dim=1) # 1, 6, 1024, 20


    if static is True:
        return feature, abs_x, idx_initial
    else:
        return feature, abs_x, idx_return




class DGCNN_Transformer(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_Transformer, self).__init__()

        self.args = args
        self.k = args.k
        self.delta = args.delta
        self.neighbors = args.neighbors
        self.rpe = args.rpe
        self.scale = args.scale

        print("self.delta : {}".format(self.delta))
        print("self.neighbors : {}".format(self.neighbors))
        print("self.rpe : {}".format(self.rpe))
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        

        self.conv1 = AttentionConv(3*2, 64, kernel_size=self.k, groups=8, rpe=self.rpe, scale=self.scale, layer=1)
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = AttentionConv(64*2, 64, kernel_size=self.k, groups=8, rpe=self.rpe, scale=self.scale, layer=2)
        self.act2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = AttentionConv(64*2, 128, kernel_size=self.k, groups=8, rpe=self.rpe, scale=self.scale, layer=3)
        self.act3 = nn.LeakyReLU(negative_slope=0.2)
        self.conv4 = AttentionConv(128*2, 256, kernel_size=self.k, groups=8, rpe=self.rpe, scale=self.scale, layer=4)
        self.act4= nn.LeakyReLU(negative_slope=0.2)

        self.conv5 = nn.Sequential(nn.Conv1d(256, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2)) # 64 + 64 + 128 + 256 = 512

        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                try:
                    if "conv" in key:
                        print(key)
                        init.kaiming_normal(self.state_dict()[key])
                except:
                    init.normal(self.state_dict()[key])
                if "bn" in key:
                    self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0




    def forward(self, x):
        batch_size = x.size(0)

        # shape of x : batch, feature, npoints, neighbors
        # convolution(shared mlp to xi, xj-xi & max)
        # =>
        # transformer(shared wq, wk, wv to xi)


        x, abs_x, idx1 = get_neighbors(x, k=self.k, delta=self.delta, neighbors=self.neighbors, layer=1) # b, 64, 1024, 20
        #print("1")
        x1 = self.conv1(x, abs_x, idx1) # b, 64, 1024
        x1 = self.act1(self.bn1(x1)).squeeze(3)

        #print("2")
        x, abs_x, idx2 = get_neighbors(x1, k=self.k, delta=self.delta, neighbors=self.neighbors, layer=2) # b, 64, 1024, 20
        x2 = self.conv2(x, abs_x, idx2) # b, 64, 1024
        x2 = self.act2(self.bn2(x2)).squeeze(3)

        #print("3")
        x, abs_x, idx3 = get_neighbors(x2, k=self.k, delta=self.delta, neighbors=self.neighbors, layer=3) # b, 64, 1024, 20
        x3 = self.conv3(x, abs_x, idx3) # b, 128, 1024
        x3 = self.act3(self.bn3(x3)).squeeze(3)

        #print("4")
        x, abs_x, idx4 = get_neighbors(x3, k=self.k, delta=self.delta, neighbors=self.neighbors, layer=4) # b, 64, 1024, 20
        x4 = self.conv4(x, abs_x, idx4) # b, 256, 1024, 20
        x4 = self.act4(self.bn4(x4)).squeeze(3)

        x = self.conv5(x4)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x



