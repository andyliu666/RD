import torch
import torch.nn as nn
import time
from DepthwiseSeparableConvolution.model_minimal import bottle_screener
#start_time = time.time()
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

        self.conv0 = nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv0 = depthwise_separable_conv(3, num_features, kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv0(x)
        return out

class bottle_screener(nn.Module):
    def __init__(self):
        super(bottle_screener, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.First_Block = First_Block(self.inplanes)
    def forward(self, x):
        x = self.First_Block(x)
        return x
'''
model = bottle_screener()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.is_available())
model.to(device)
model.eval()

input = torch.randn((1,3,224,224)).to(device)
input = torch.mul(input, 2.2).to(device)
print(input.size())

out = model(input)
'''