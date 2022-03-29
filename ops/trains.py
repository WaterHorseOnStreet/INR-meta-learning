import os
import time
import copy
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.data import Mixup
import ops.meters as meters

import models

def train(model, optimizer, 
            train_set, val_set, 
            train_scheduler, warmup_scheduler, gpu,
            writer=None, save_dict="model_checkpoints",uid=None,
            verbose=1):
    
    warmup_epochs = 5
    model = model.cuda() if gpu else model.cpu()

    warmup_time = time.time()

    for epoch in range(warmup_epochs):
        batch_time = time.time()
        train_metrics = train_epoch(model, optimizer, train_set, scheduler=warmup_scheduler, gpu=gpu)

        batch_time = time.time() - batch_time
        template = "(%.2f sec/epoch) Warmup epoch: %d, Loss: %.4f, lr: %.3e"
        print(template % (batch_time,
                          epoch,
                          train_metrics,
                          [param_group["lr"] for param_group in optimizer.param_groups][0]))

    if warmup_epochs > 0:
        print("The model is warmed up: %.2f sec \n" % (time.time() - warmup_time))

    epochs = 50
    for epoch in range(epochs):
            batch_time = time.time()
            train_metrics = train_epoch(model,optimizer,train_set,scheduler=train_scheduler,
                                        gpu=gpu)
            train_scheduler.step()
            batch_time = time.time() - batch_time

            # if writer is not None and (epoch + 1) % 1 == 0:
                # add_train_metrics(writer, train_metrics, epoch)
            template = "(%.2f sec/epoch) Epoch: %d, Loss: %.4f, lr: %.3e"
            print(template % (batch_time,
                            epoch,
                            train_metrics,
                            [param_group["lr"] for param_group in optimizer.param_groups][0]))

def train_epoch(model, optimizer, train_set,
                scheduler=None, gpu=True):
    model.train()
    loss_function = nn.CrossEntropyLoss()

    loss_function = loss_function.cuda() if gpu else loss_function

    loss_meter=meters.AverageMeter("loss")

    domain_weight=0.8

    for step, (xs, cls, domain, idx) in enumerate(train_set):
        if gpu:
            xs = xs.cuda()
            cls = cls.cuda()
            domain = domain.cuda()

        optimizer.zero_grad()
        logits = model(xs)
        class_pred = logits[:,:65]
        domain_pred = logits[:,65:]
        loss_class = loss_function(class_pred,cls)
        loss_domain = loss_function(domain_pred, domain)

        loss = (1-domain_weight)*loss_class + domain_weight*loss_domain
        

        loss_meter.update(loss.item())

        loss.backward()

        optimizer.step()

        if scheduler:
            scheduler.step()

    return loss_meter.avg


