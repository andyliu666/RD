# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch.nn as nn

from .unet_parts import *

class UNet_extra1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_extra1, self).__init__()
        self.inc = inconv(n_channels, 64) # 512*64
        self.down1 = down(64, 128) # 256*32
        self.down2 = down(128, 256) # 128*16
        self.down3 = down(256, 512) # 64*8
        self.down4 = down(512, 512) # 32*4
        self.up1 = up(1024, 256, bilinear=bilinear)
        self.up2 = up(512, 128, bilinear=bilinear)
        self.up3 = up(256, 64, bilinear=bilinear)
        self.up4 = up(128, 64, bilinear=bilinear)
        self.outc = outconv(64, n_classes)
        self.extra_global = nn.Linear(512*32*4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
#         return F.sigmoid(x)
        if self.training:
           x5_flatten = x5.view(x5.shape[0], -1)
           #print(x5_flatten.shape)
           extra_1 = self.extra_global(x5_flatten)
           return x, extra_1
        else:
           return x

    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, bilinear=bilinear)
        self.up2 = up(512, 128, bilinear=bilinear)
        self.up3 = up(256, 64, bilinear=bilinear)
        self.up4 = up(128, 64, bilinear=bilinear)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)
        #return x


class UNet_multi(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_multi1, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, bilinear=bilinear)
        
        self.up2 = up(512, 128, bilinear=bilinear)
        #self.up2_m = nn.ConvTranspose2d(128, 64, 4, stride=4)
        #self.outc_2_m = outconv(64, n_classes)
        
        self.up3 = up(256, 64, bilinear=bilinear)
        self.up3_m = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.outc_3_m = outconv(64, n_classes)
        
        self.up4 = up(128, 64, bilinear=bilinear)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4_ = self.up1(x5, x4)
        x3_ = self.up2(x4_, x3)
        x2_ = self.up3(x3_, x2)
        x1_ = self.up4(x2_, x1)
        x = self.outc(x1_)

        if self.training:
            x2_m = self.up3_m(x2_)
            x2_m = self.outc_3_m(x2_m)
            
            #x3_m = self.up2_m(x3_)
            #x3_m = self.outc_2_m(x3_m)
            
            return x, x2_m
        else:
            x2_m = self.up3_m(x2_)
            x2_m = self.outc_3_m(x2_m)
            
            #x3_m = self.up2_m(x3_)
            #x3_m = self.outc_2_m(x3_m)
            
            x = F.softmax(x)
            x2_m = F.softmax(x2_m)
            #x3_m = F.softmax(x3_m)
            
            return 0.5 * x + 0.5 * x2_m


class UNet_multi1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_multi1, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, bilinear=bilinear)
        
        self.up2 = up(512, 128, bilinear=bilinear)
        #self.up2_m = nn.ConvTranspose2d(128, 64, 4, stride=4)
        #self.outc_2_m = outconv(64, n_classes)
        
        self.up3 = up(256, 64, bilinear=bilinear)
        self.up3_m = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.outc_3_m = outconv(64, n_classes)
        
        self.up4 = up(128, 64, bilinear=bilinear)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4_ = self.up1(x5, x4)
        x3_ = self.up2(x4_, x3)
        x2_ = self.up3(x3_, x2)
        x1_ = self.up4(x2_, x1)
        x = self.outc(x1_)

        if self.training:
            x2_m = self.up3_m(x2_)
            x2_m = self.outc_3_m(x2_m)
            
            #x3_m = self.up2_m(x3_)
            #x3_m = self.outc_2_m(x3_m)
            
            return x, x2_m
        else:
            x2_m = self.up3_m(x2_)
            x2_m = self.outc_3_m(x2_m)
            
            #x3_m = self.up2_m(x3_)
            #x3_m = self.outc_2_m(x3_m)
            
            x = F.softmax(x)
            x2_m = F.softmax(x2_m)
            #x3_m = F.softmax(x3_m)
            
            return 0.5 * x + 0.5 * x2_m
        
        
class UNet_multi2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_multi2, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, bilinear=bilinear)
        
        self.up2 = up(512, 128, bilinear=bilinear)
        self.up2_m = nn.ConvTranspose2d(128, 64, 4, stride=4)
        self.outc_2_m = outconv(64, n_classes)
        
        self.up3 = up(256, 64, bilinear=bilinear)
        self.up3_m = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.outc_3_m = outconv(64, n_classes)
        
        self.up4 = up(128, 64, bilinear=bilinear)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4_ = self.up1(x5, x4)
        x3_ = self.up2(x4_, x3)
        x2_ = self.up3(x3_, x2)
        x1_ = self.up4(x2_, x1)
        x = self.outc(x1_)

        if self.training:
            x2_m = self.up3_m(x2_)
            x2_m = self.outc_3_m(x2_m)
            
            x3_m = self.up2_m(x3_)
            x3_m = self.outc_2_m(x3_m)
            
            return x, x2_m, x3_m
        else:
            x2_m = self.up3_m(x2_)
            x2_m = self.outc_3_m(x2_m)
            
            x3_m = self.up2_m(x3_)
            x3_m = self.outc_2_m(x3_m)
            
            x = F.softmax(x)
            x2_m = F.softmax(x2_m)
            x3_m = F.softmax(x3_m)
            
            return 0.6 * x + 0.2 * x2_m + 0.2 * x3_m
        

class UNet_7x3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_7x3, self).__init__()
        self.inc = inconv_7x3(n_channels, 64)
        self.down1 = down_7x3(64, 128)
        self.down2 = down_7x3(128, 256)
        self.down3 = down_7x3(256, 512)
        self.down4 = down_7x3(512, 512)
        self.up1 = up_7x3(1024, 256, bilinear=bilinear)
        self.up2 = up_7x3(512, 128, bilinear=bilinear)
        self.up3 = up_7x3(256, 64, bilinear=bilinear)
        self.up4 = up_7x3(128, 64, bilinear=bilinear)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
#         return F.sigmoid(x)
        return x


class UNet_small(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_small, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 64)
        self.down2 = down(64, 64)
        self.down3 = down(64, 64)
        self.down4 = down(64, 64)
        self.up1 = up(128, 64, bilinear=bilinear)
        self.up2 = up(128, 64, bilinear=bilinear)
        self.up3 = up(128, 64, bilinear=bilinear)
        self.up4 = up(128, 64, bilinear=bilinear)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
#         return F.sigmoid(x)
        return x


class UNet_SE(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_SE, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.se1 = ChannelSELayer(64)
        self.down1 = down(64, 64)
        self.se2 = ChannelSELayer(64)
        self.down2 = down(64, 64)
        self.se3 = ChannelSELayer(64)
        self.down3 = down(64, 64)
        self.se4 = ChannelSELayer(64)
        self.down4 = down(64, 64)
        self.up1 = up(128, 64, bilinear=bilinear)
        self.se1u = ChannelSELayer(64)
        self.up2 = up(128, 64, bilinear=bilinear)
        self.se2u = ChannelSELayer(64)
        self.up3 = up(128, 64, bilinear=bilinear)
        self.se3u = ChannelSELayer(64)
        self.up4 = up(128, 64, bilinear=bilinear)
        self.se4u = ChannelSELayer(64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.se1(x1)
        x2 = self.down1(x1)
        x1 = self.se2(x2)
        x3 = self.down2(x2)
        x3 = self.se3(x3)
        x4 = self.down3(x3)
        x4 = self.se4(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.se1u(x)
        x = self.up2(x, x3)
        x = self.se2u(x)
        x = self.up3(x, x2)
        x = self.se3u(x)
        x = self.up4(x, x1)
        x = self.se4u(x)
        x = self.outc(x)
#         return F.sigmoid(x)
        return x