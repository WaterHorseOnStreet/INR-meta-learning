import io
import time
import csv
from turtle import mode

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import ops.meters as meters

@torch.no_grad()
def test(model, val_set, transform=None, verbose=False, gpu=True, writer=None):
    model.eval()

    model = model.cuda() if gpu else model.cpu()


    count = 1
    acc = 0
    for step, (xs, cls, domain, idx) in enumerate(val_set):
        if gpu:
            xs = xs.cuda()
            cls = cls.cuda()
        if transform is not None:
            xs, cls_t = transform(xs, cls)
        else:
            xs, cls_t = xs, cls   

        pred = model(xs)
        cls_pred = F.softmax(pred[:,:65])  

        cls_t = cls_t.cpu()
        cls = cls.cpu()
        cls_pred = cls_pred.cpu() 

        acc += calc_accuracy(xs, cls_pred, cls_t)


        count = count + 1

    print("test accuracy is {}".format(acc/count))
    return acc/count


def calc_accuracy(xs, pred, ys):
    # reduce/collapse the classification dimension according to max op
    # resulting in most likely label
    _, cls_pred = torch.max(pred.data, dim=1)
    # assumes the first dimension is batch size
    n = xs.size(0)  # index 0 for extracting the # of elements
    # calulate acc (note .item() to do float division)
    acc = (cls_pred == ys).sum().item() / n
    return acc