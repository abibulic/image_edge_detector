import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as resnet

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class ResNetHed(nn.Module):
    def __init__(self, args):
        super(ResNetHed, self).__init__()

        self.hparams = args
                        
        ch_num = 64
        if '18' in self.hparams.backbone_model:
            base_model = resnet.resnet18(pretrained=self.hparams.pretrained)
        elif '34' in self.hparams.backbone_model:
            base_model = resnet.resnet34(pretrained=self.hparams.pretrained)
            
        base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
    
        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        self.conv_original_size0 = convrelu(self.hparams.n_channels, ch_num, 3, 1)
        self.conv_original_size1 = convrelu(ch_num, ch_num, 3, 1)
    
        self.score_dsn_original = nn.Conv2d(64, 1, 1)
        self.score_dsn0 = nn.Conv2d(64, 1, 1)
        self.score_dsn1 = nn.Conv2d(64, 1, 1)
        self.score_dsn2 = nn.Conv2d(128, 1, 1)
        self.score_dsn3 = nn.Conv2d(256, 1, 1)
        self.score_dsn4 = nn.Conv2d(512, 1, 1)

        self.score_final1 = convrelu(5, 5, 3, 1)
        self.score_final2 = nn.Conv2d(5, 1, 1)


    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        #layer4 = self.layer4(layer3)

        res_original = self.score_dsn_original(x_original)
        res0 = self.score_dsn0(layer0)
        res1 = self.score_dsn1(layer1)
        res2 = self.score_dsn2(layer2)
        res3 = self.score_dsn3(layer3)
        #res4 = self.score_dsn4(layer4)

        res00 = self.upsample0(res0)
        res11 = self.upsample1(res1)
        res22 = self.upsample2(res2)
        res33 = self.upsample3(res3)
        #res4 = self.upsample4(res4)

        fusecat = torch.cat((res_original, res00, res11, res22, res33), dim=1)
        #fuse = self.score_final1(fusecat)
        fuse = self.score_final2(fusecat)
        results = [res_original, res00, res11, res22, res33, fuse]
        results = [torch.sigmoid(r) for r in results]

        if self.hparams.create_onnx:
            return results[-1]
        else:
            return results