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

        self.conv_original_size0 = convrelu(self.hparams.n_channels, ch_num, 3, 1)
        self.conv_original_size1 = convrelu(ch_num, ch_num, 3, 1)

        self.down1 = nn.Sequential(*base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.down2 = nn.Sequential(*base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.down3 = base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.down4 = base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
    
        self.up1_4 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.up1_3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.up1_2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.up1_1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)

        self.up2_3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.up2_2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.up2_1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)

        self.up3_2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.up3_1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)

        self.up4_1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)

        self.vote1_4 = convrelu(256, 128, 3, 1)
        self.vote1_3 = convrelu(128, 64, 3, 1)
        self.vote1_2 = convrelu(128, 64, 3, 1)
        self.vote1_1 = convrelu(128, 64, 3, 1)

        self.vote2_3 = convrelu(128, 64, 3, 1)
        self.vote2_2 = convrelu(128, 64, 3, 1)
        self.vote2_1 = convrelu(128, 64, 3, 1)

        self.vote3_2 = convrelu(128, 64, 3, 1)
        self.vote3_1 = convrelu(128, 64, 3, 1)

        self.vote4_1 = convrelu(128, 64, 3, 1)
    
        self.score_original = nn.Conv2d(64, 1, 1)
        self.score1 = nn.Conv2d(64, 1, 1)
        self.score2 = nn.Conv2d(64, 1, 1)
        self.score3 = nn.Conv2d(64, 1, 1)
        self.score4 = nn.Conv2d(64, 1, 1)

        self.score_final = nn.Conv2d(5, 1, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        res_original = self.score_original(x_original)

        #DOWN LAYER
        down1 = self.down1(input)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        #UP LAYER 1
        up1_4_1 = self.up1_4(down4)
        up1_4_2 = torch.cat((down3, up1_4_1), dim=1)
        up1_4_3 = self.vote1_4(up1_4_2)

        up1_3_1 = self.up1_3(up1_4_3)
        up1_3_2 = torch.cat((down2, up1_3_1), dim=1)
        up1_3_3 = self.vote1_3(up1_3_2)

        up1_2_1 = self.up1_2(up1_3_3)
        up1_2_2 = torch.cat((down1, up1_2_1), dim=1)
        up1_2_3 = self.vote1_2(up1_2_2)

        up1_1_1 = self.up1_1(up1_2_3)
        up1_1_2 = torch.cat((x_original, up1_1_1), dim=1)
        up1_1_3 = self.vote1_1(up1_1_2)

        res1 = self.score1(up1_1_3)

        #UP LAYER 2
        up2_3_1 = self.up2_3(down3)
        up2_3_2 = torch.cat((down2, up2_3_1), dim=1)
        up2_3_3 = self.vote2_3(up2_3_2)

        up2_2_1 = self.up2_2(up2_3_3)
        up2_2_2 = torch.cat((down1, up2_2_1), dim=1)
        up2_2_3 = self.vote2_2(up2_2_2)

        up2_1_1 = self.up2_1(up2_2_3)
        up2_1_2 = torch.cat((x_original, up2_1_1), dim=1)
        up2_1_3 = self.vote2_1(up2_1_2)

        res2 = self.score2(up2_1_3)

        #UP LAYER 3
        up3_2_1 = self.up3_2(down2)
        up3_2_2 = torch.cat((down1, up3_2_1), dim=1)
        up3_2_3 = self.vote3_2(up3_2_2)

        up3_1_1 = self.up3_1(up3_2_3)
        up3_1_2 = torch.cat((x_original, up3_1_1), dim=1)
        up3_1_3 = self.vote3_1(up3_1_2)

        res3 = self.score2(up3_1_3)

        #UP LAYER 4
        up4_1_1 = self.up4_1(down1)
        up4_1_2 = torch.cat((x_original, up4_1_1), dim=1)
        up4_1_3 = self.vote3_1(up4_1_2)

        res4 = self.score2(up4_1_3)
   
        res_cat = torch.cat((res_original, res1, res2, res3, res4), dim=1)

        fuse = self.score_final(res_cat)

        results = [res_original, res1, res2, res3, res4, fuse]
        results = [torch.sigmoid(r) for r in results]

        if self.hparams.create_onnx:
            return results[-1]
        else:
            return results