import os
import random
from xml import dom
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from dataio.base import OfficeHomeDataset
from dataio.datasets import get_HomeOfficeDataloader
import models
import torch
import torch.nn as nn
import torch.optim as optim
import ops.schedulers as schedulers
import ops.trains as trains
import datetime
import ops.tests as tests

here = os.getcwd()
dataset_path = os.path.join(here,"./OfficeHomeDataset")

def main():
    source_domain = ['Art','Clipart','Product']
    target_domain = ['Real_World']

    train_set, target_set, num_classes = get_HomeOfficeDataloader(dataset_path, source_domain, target_domain)

    name = "resnet_dnn_18"

    vit_kwargs = {
    "image_size": 32, 
    "patch_size": 2,
    }

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model = models.get_model(name, num_classes=num_classes, 
                            stem=True, **vit_kwargs)
                            
    checkpoint = torch.load('./models_checkpoints/office_home/resnet_dnn_18/office_home_resnet_dnn_18_20220324_111939.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    

    name = model.name
    model = nn.DataParallel(model)
    model.name = name
    gpu = torch.cuda.is_available()

    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    train_scheduler = optim.lr_scheduler.StepLR(optimizer,0.01)
    warmup_scheduler = schedulers.WarmupScheduler(optimizer, len(train_set) * 5)

    # trains.train(model, optimizer,
    #                 train_set, target_set,
    #                 train_scheduler, warmup_scheduler,
    #                 gpu=gpu,uid=current_time)

    # models.save(model, "office_home", current_time, optimizer=optimizer)

    # print("training is done!!")


    tests.test(model,target_set)


if __name__ == "__main__":
    main()