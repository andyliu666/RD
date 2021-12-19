import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# from hwcounter import count, count_end

from torch.quantization import QuantStub, DeQuantStub

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class First_Block(nn.Module):
    def __init__(self, num_features):
        super(First_Block, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.norm0 = nn.BatchNorm2d(num_features)
        self.norm1 = nn.BatchNorm2d(num_features)
        self.norm2 = nn.BatchNorm2d(num_features)

        
        self.conv0 = depthwise_separable_conv(3, num_features, kernel_size=3, padding=1, bias=False)
        self.conv1 = depthwise_separable_conv(num_features, num_features, kernel_size=3, padding=1, bias=False)
        self.conv2 = depthwise_separable_conv(num_features, num_features, kernel_size=3, padding=1, bias=False)
        # self.conv0 = nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        #print("Model State:", self.training)
        #Might want to use the above to switch between C implementations and python ones

        # start = count()
        out = self.conv0(x)
        # elapsed = count_end() - start
        #print("conv ,", elapsed, ",",out.size())
        #print("in:",out[0][0][0][0])
        # start = count()
        out = self.norm0(out)
        # elapsed = count_end() - start
        #print("norm ,", elapsed, ",",out.size())
        # start = count()
        out = self.relu(out)
        # elapsed = count_end() - start
        #print("relu ,", elapsed, ",",out.size())
        # start = count()
        out = self.conv1(out)
        # elapsed = count_end() - start
        #print("conv ,", elapsed, ",",out.size())
        # start = count()
        out = self.norm1(out)
        # elapsed = count_end() - start
        #print("norm ,", elapsed, ",",out.size())
        # start = count()
        out = self.relu(out)
        # elapsed = count_end() - start
        #print("relu ,", elapsed, ",",out.size())
        # start = count()
        out = self.conv2(out)
        # elapsed = count_end() - start
        #print("conv ,", elapsed, ",",out.size())
        # start = count()
        out = self.norm2(out)
        # elapsed = count_end() - start
        #print("norm ,", elapsed, ",",out.size())
        # start = count()
        out = self.relu(out)
        # elapsed = count_end() - start
        #print("relu ,", elapsed, ",",out.size())
        # start = count()
        out = self.pooling(out)
        # elapsed = count_end() - start
        #print("max pooling,", elapsed, ",",out.size())

        return out


class Add_Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Add_Block, self).__init__()
        self.conv1 = depthwise_separable_conv(inplanes, planes, kernel_size=3, padding=1, bias=False)
        #self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, dilation=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = depthwise_separable_conv(planes, planes, kernel_size=3, padding=1, bias=False)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
    #    print("ad block top")
    #    print(x.size())
        # start = count()
        out = self.conv1(x)
        # elapsed = count_end() - start
        #print("conv ,", elapsed, ",",out.size())
    #    print(out.size())
        # start = count()
        out = self.bn1(out)
        # elapsed = count_end() - start
        #print("bn ,", elapsed, ",",out.size())
        # start = count()
        out = self.relu(out)
        # elapsed = count_end() - start
        #print("relu ,", elapsed, ",",out.size())
    #    print(out.size())
        # start = count()
        out = self.conv2(out)
        # elapsed = count_end() - start
        #print("conv ,", elapsed, ",",out.size())
    #    print(out.size())
        # start = count()
        out = self.bn2(out)
        # elapsed = count_end() - start
        #print("bn ,", elapsed, ",",out.size())
    #    print(out.size())
        # start = count()
        if self.downsample is not None:
            identity = self.downsample(x)
        #out += identity
        # elapsed = count_end() - start
        #print("add layer , ", elapsed, ",", out.size())

        # start = count()
        out = self.relu(out)
        # elapsed = count_end() - start
        #print("relu ,", elapsed, ",",out.size())
        return out

class Concat_Layer(nn.Module):
    def __init__(self, num_features, growth_rate, bn_size):
        super(Concat_Layer, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.norm0 = nn.BatchNorm2d(num_features)
        self.norm1 = nn.BatchNorm2d(bn_size * growth_rate)

        self.conv0 = depthwise_separable_conv(num_features, bn_size * growth_rate, kernel_size=1, padding=0, bias=False)
        self.conv1 = depthwise_separable_conv(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        # self.conv0 = nn.Conv2d(num_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        # self.conv1 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        #print("concat layer top")
        # start = count()
        out = self.norm0(x)
        # elapsed = count_end() - start
        #print("norm ,", elapsed, ",",out.size())
        #print(out.size())
        # start = count()
        out = self.relu(out)
        # elapsed = count_end() - start
        #print("relu ,", elapsed, ",",out.size())
        #print(out.size())
        # start = count()
        out = self.conv0(out)
        # elapsed = count_end() - start
        #print("conv ,", elapsed, ",",out.size())
        #print(out.size())
        # start = count()
        out = self.norm1(out)
        # elapsed = count_end() - start
        #print("norm ,", elapsed, ",",out.size())
        #print(out.size())
        # start = count()
        out = self.relu(out)
        # elapsed = count_end() - start
        #print("relu ,", elapsed, ",",out.size())
        #print(out.size())
        # start = count()
        out = self.conv1(out)
        # elapsed = count_end() - start
        #print("conv ,", elapsed, ",",out.size())
        #print(out.size(), x.size())
        out_concat = torch.cat([x, out], 1)
        #print(out_concat.size())
        #print("concat layer bottom")
        return out_concat


class Concat_Block(nn.Sequential):
    def __init__(self, num_layers, num_features, bn_size, growth_rate):
        super(Concat_Block, self).__init__()
        for i in range(num_layers):
            layer = Concat_Layer(num_features + i * growth_rate, growth_rate,
                                bn_size)
            self.add_module('Concat_Layer%d' % (i + 1), layer)

class Shift_Layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(Shift_Layer, self).__init__()

        self.conv = depthwise_separable_conv(in_features, out_features, kernel_size=1, bias=False)
        #self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.norm = nn.BatchNorm2d(in_features)

    def forward(self, x):
        #print("shift top")
        #print()
        # start = count()
        
        out = self.norm(x)
        # elapsed = count_end() - start
        #print("norm ,", elapsed, ",",out.size())
        #print(out.size())
        # start = count()
        
        out = self.relu(out)
        # elapsed = count_end() - start
        #print("relu ,", elapsed, ",",out.size())
        #print(out.size())
        # start = count()
        
        out = self.conv(out)
        # elapsed = count_end() - start
        #print("conv ,", elapsed, ",",out.size())
        #print(out.size())
        # start = count()
        
        out = self.pooling(out)
        # elapsed = count_end() - start
        #print("avg pooling,", elapsed, ",",out.size())
        #print(out.size())
        #print("shift bottom")
        return out


class bottle_screener(nn.Module):
    def __init__(self, block = Add_Block, layers = [6, 2, 6, 2], num_classes=2):
        super(bottle_screener, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.First_Block = First_Block(self.inplanes)
        self.Block1 = self.Create_Concat_Block(64, layers[0])
        self.Block2 = self.Create_Add_Block(block, 256, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def Create_Add_Block(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                depthwise_separable_conv(self.inplanes, planes, kernel_size=1, bias=False),
                #nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def Create_Concat_Block(self, planes, blocks, growth_rate = 32, bn_size = 4):
        layers = []
        num_features = self.inplanes
        layers.append(Concat_Block(num_layers=blocks, num_features=num_features,bn_size=bn_size, growth_rate=growth_rate))
        num_features = num_features + blocks * growth_rate
        layers.append(Shift_Layer(in_features=num_features, out_features=num_features // 2))
        self.inplanes = num_features // 2
        return nn.Sequential(*layers)


    def forward(self, x):
        #print("model num_threads",torch.get_num_threads())
        # print("smaller model")
        x_in = x
        x = self.quant(x)
        x = self.First_Block(x)
        # print(x.size())
        #print("\n")
        x = self.Block1(x)
        # print(x.size())
        #print("\n")
        x = self.Block2(x)
        # print(x.size())
        #start = count()
        x = self.avgpool(x)
        #print(x.size())
        #elapsed = count_end() - start
        #print("avgpool ," , elapsed, ",", x.size())
        #print("\n")
        x = x.reshape(x.size(0), -1)
        #start = count()
        x = self.fc(x)

        x = self.dequant(x)
        #elapsed = count_end() - start
        #print("fc ," , elapsed, ",", x.size())

        return x
