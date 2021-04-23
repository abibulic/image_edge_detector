import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from dataset import EdgeDataset
from vgg import VggHed, load_vgg16pretrain
from resnet_modified3 import ResNetHed


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def list_files(root_dir, ext='.jpg'):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(ext):
                file_list.append(os.path.join(root, file).replace("\\","/"))
    return file_list

def vgg_weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()


class HED(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.hparams = args

        if 'vgg' in self.hparams.backbone_model:
            self.model = VggHed(args)
            self.model.cuda()
            self.model.apply(vgg_weights_init)
            if self.hparams.pretrained:
                load_vgg16pretrain(self.model)

        else:
            self.model = ResNetHed(self.hparams)    

            #freez resnet
            # for name, param in self.model.named_parameters():
            #     if 'down' in name or 'original_size' in name:
            #         param.requires_grad = False

    def forward(self, input):
        return self.model(input)

        
    def prepare_data(self):
        #TODO this part need some rearrangement
        files = list_files(self.hparams.data_path, ext='.json')
        files = files

        files_BIPED_train = list_files(f'{self.hparams.disc[1]}:/OpenDataSources/EdgeDetection/BIPED/edges/imgs/train/rgbr/real/', ext='.jpg')
        files_BIPED_test = list_files(f'{self.hparams.disc[1]}:/OpenDataSources/EdgeDetection/BIPED/edges/imgs/test/rgbr/', ext='.jpg')

        files_exit = list_files(f'{self.hparams.disc[1]}:/LocalDataSources/EdgeDetection/exit/json/', ext='.json')
        files_exit = 4*files_exit

        f = open(f'{self.hparams.disc[1]}:/OpenDataSources/EdgeDetection/HED-BSDS/train_pair.lst', 'r')
        files_BSDS = f.readlines()
        f.close()

        files = files# + files_exit# + files_BSDS

        train_data, valid_data = train_test_split(files, test_size=self.hparams.valid_split, random_state=0, shuffle=True)

        self.data_train = EdgeDataset(train_data, self.hparams, transforms=True)
        self.data_valid = EdgeDataset(valid_data, self.hparams, transforms=False)


    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.data_train,
                                               batch_size=self.hparams.batch_size, shuffle=True,
                                               num_workers=self.hparams.num_workers, pin_memory=False)
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(self.data_valid,
                                               batch_size=self.hparams.batch_size,
                                               num_workers=self.hparams.num_workers, pin_memory=False)
        return valid_loader

    def configure_optimizers(self):
        #TODO treba postaviti lr za svaki dio posebno
        if 'vgg' in self.hparams.backbone_model:
            net_parameters_id = {}
            net = self.model
            for pname, p in net.named_parameters():
                if pname in ['conv1_1.weight','conv1_2.weight',
                            'conv2_1.weight','conv2_2.weight',
                            'conv3_1.weight','conv3_2.weight','conv3_3.weight',
                            'conv4_1.weight','conv4_2.weight','conv4_3.weight']:
                    print(pname, 'lr:1 de:1')
                    if 'conv1-4.weight' not in net_parameters_id:
                        net_parameters_id['conv1-4.weight'] = []
                    net_parameters_id['conv1-4.weight'].append(p)
                elif pname in ['conv1_1.bias','conv1_2.bias',
                            'conv2_1.bias','conv2_2.bias',
                            'conv3_1.bias','conv3_2.bias','conv3_3.bias',
                            'conv4_1.bias','conv4_2.bias','conv4_3.bias']:
                    print(pname, 'lr:2 de:0')
                    if 'conv1-4.bias' not in net_parameters_id:
                        net_parameters_id['conv1-4.bias'] = []
                    net_parameters_id['conv1-4.bias'].append(p)
                elif pname in ['conv5_1.weight','conv5_2.weight','conv5_3.weight']:
                    print(pname, 'lr:100 de:1')
                    if 'conv5.weight' not in net_parameters_id:
                        net_parameters_id['conv5.weight'] = []
                    net_parameters_id['conv5.weight'].append(p)
                elif pname in ['conv5_1.bias','conv5_2.bias','conv5_3.bias'] :
                    print(pname, 'lr:200 de:0')
                    if 'conv5.bias' not in net_parameters_id:
                        net_parameters_id['conv5.bias'] = []
                    net_parameters_id['conv5.bias'].append(p)

                elif pname in ['score_dsn1.weight','score_dsn2.weight','score_dsn3.weight',
                            'score_dsn4.weight','score_dsn5.weight']:
                    print(pname, 'lr:0.01 de:1')
                    if 'score_dsn_1-5.weight' not in net_parameters_id:
                        net_parameters_id['score_dsn_1-5.weight'] = []
                    net_parameters_id['score_dsn_1-5.weight'].append(p)
                elif pname in ['score_dsn1.bias','score_dsn2.bias','score_dsn3.bias',
                            'score_dsn4.bias','score_dsn5.bias']:
                    print(pname, 'lr:0.02 de:0')
                    if 'score_dsn_1-5.bias' not in net_parameters_id:
                        net_parameters_id['score_dsn_1-5.bias'] = []
                    net_parameters_id['score_dsn_1-5.bias'].append(p)
                elif pname in ['score_final.weight']:
                    print(pname, 'lr:0.001 de:1')
                    if 'score_final.weight' not in net_parameters_id:
                        net_parameters_id['score_final.weight'] = []
                    net_parameters_id['score_final.weight'].append(p)
                elif pname in ['score_final.bias']:
                    print(pname, 'lr:0.002 de:0')
                    if 'score_final.bias' not in net_parameters_id:
                        net_parameters_id['score_final.bias'] = []
                    net_parameters_id['score_final.bias'].append(p)

            optimizer = torch.optim.SGD([
                    {'params': net_parameters_id['conv1-4.weight']      , 'lr': self.hparams.lr*1    , 'weight_decay': self.hparams.weight_decay},
                    {'params': net_parameters_id['conv1-4.bias']        , 'lr': self.hparams.lr*2    , 'weight_decay': 0.},
                    {'params': net_parameters_id['conv5.weight']        , 'lr': self.hparams.lr*100  , 'weight_decay': self.hparams.weight_decay},
                    {'params': net_parameters_id['conv5.bias']          , 'lr': self.hparams.lr*200  , 'weight_decay': 0.},
                    {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': self.hparams.lr*0.01 , 'weight_decay': self.hparams.weight_decay},
                    {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': self.hparams.lr*0.02 , 'weight_decay': 0.},
                    {'params': net_parameters_id['score_final.weight']  , 'lr': self.hparams.lr*0.001, 'weight_decay': self.hparams.weight_decay},
                    {'params': net_parameters_id['score_final.bias']    , 'lr': self.hparams.lr*0.002, 'weight_decay': 0.},
                ], lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, verbose=True, patience=10)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        else:
            grouped_parameters = []
            params = list(self.named_parameters())
            for param in params:
                if 'original' in param[0]:
                    grouped_parameters.append({"params": param[1], 'lr': self.hparams.lr*5})
                elif 'layer0' in param[0] or 'dsn0' in param[0]:
                    grouped_parameters.append({"params": param[1], 'lr': self.hparams.lr*3})
                elif 'layer1' in param[0] or 'dsn1' in param[0]:
                    grouped_parameters.append({"params": param[1], 'lr': self.hparams.lr*2})
                elif 'layer2' in param[0] or 'dsn2' in param[0]:
                    grouped_parameters.append({"params": param[1], 'lr': self.hparams.lr*1})
                elif 'layer3' in param[0] or 'dsn3' in param[0]:
                    grouped_parameters.append({"params": param[1], 'lr': self.hparams.lr*0.1})
                elif 'layer4' in param[0] or 'dsn4' in param[0]:
                    grouped_parameters.append({"params": param[1], 'lr': self.hparams.lr*0.01})
                else:
                    grouped_parameters.append({"params": param[1], 'lr': self.hparams.lr})

            #optimizer = torch.optim.SGD(grouped_parameters, lr=self.hparams.lr)
            #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, verbose=True, patience=10)

        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val_loss'
            }
        
    def loss_function(self, pred, target, gauss):
        loss = torch.zeros(1).cuda()
        target = target.long()
        mask1 = (target != 0).float()
        num_positive = torch.sum(mask1).float()
        num_negative = mask1.numel() - num_positive
        mask1[mask1 != 0] = num_negative / (num_positive + num_negative)
        mask1[mask1 == 0] = num_positive / (num_positive + num_negative)

        mask2 = mask1.clone()
        mask2 += gauss
        #mask2 = (mask2)/(torch.max(mask2) - torch.min(mask2))

        #### 1
        # for i, p in enumerate(pred):
        #     if i < 5:
        #         cost = F.binary_cross_entropy(p.float(), target.float(), weight=mask1, reduction='none')
        #         weight_factor = 0.05*(i+1)
        #         loss += weight_factor*torch.sum(cost)
        #     else:
        #         cost = F.binary_cross_entropy(p.float(), target.float(), weight=mask2, reduction='none')
        #         loss += torch.sum(cost)

        #### 2
        sum_res = torch.zeros(target.shape).cuda()
        for i, p in enumerate(pred):
            sum_res += p
        sum_res /= 6 
           
        #### 3
        # p = pred[-1]

        loss = torch.sum(F.binary_cross_entropy(sum_res.float(), target.float(), weight=mask2, reduction='none'))
        return loss
    
    def calc_F1(self, pred, target, threshold):
        p = pred[-1].clone()
        p[p < threshold] = 0
        p[p >= threshold] = 1

        tp = (target * p).sum().to(torch.float32)
        tn = ((1 - target) * (1 - p)).sum().to(torch.float32)
        fp = ((1 - target) * p).sum().to(torch.float32)
        fn = (target * (1 - p)).sum().to(torch.float32)

        epsilon = 1e-7

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        
        f1 = 2* (precision*recall) / (precision + recall + epsilon)

        return 1-f1

    def valid_function(self, pred, target):
        p = pred[-1].clone()
        p[p < 0.5] = 0
        p[p >= 0.5] = 1
        loss = torch.sum(torch.abs(p-target))
        return loss

    def training_step(self, batch, batch_idx):
        x, y, g, _ = batch
        outputs = self.forward(x)
        loss = self.loss_function(outputs, y, g)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, g, _ = batch
        outputs = self.forward(x)
        loss = self.calc_F1(outputs, y, 0.5)
        self.log("val_loss", loss, on_epoch=True) 
        