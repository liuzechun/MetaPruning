import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math

import numpy as np

channel_scale = []
for i in range(31):
    channel_scale += [(10 + i * 3)/100]

stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2

rngs = [28, 18, 10, 23, 23, 22, 22, 13, 17, 28, 17, 25, 29]

channel = []
for i in range(len(stage_out_channel)):
    if i == 13:
        channel += [stage_out_channel[i]]
    else:
        channel += [int(stage_out_channel[i] * channel_scale[rngs[i-1]])]

class conv_3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(conv_3x3, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        return out

class dw3x3_pw1x1(nn.Module):
    def __init__(self, inp, oup, stride):
        super(dw3x3_pw1x1, self).__init__()

        self.dwconv_3x3 = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.pwconv_1x1 = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)

    def forward(self, x):

        out = self.dwconv_3x3(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.pwconv_1x1(out)
        out = self.bn2(out)
        out = F.relu(out)

        return out

class MobileNetV1(nn.Module):
    def __init__(self, input_size=224, num_classes=1000):
        super(MobileNetV1, self).__init__()


        self.feature = nn.ModuleList()
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(conv_3x3(3, channel[i], 2))
            elif stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != 64:
                self.feature.append(dw3x3_pw1x1(channel[i-1], channel[i], 2))
            else:
                self.feature.append(dw3x3_pw1x1(channel[i-1], channel[i], 1))

        #self.feature.append(dw3x3_pw1x1(32, 64, 1))
        #self.feature.append(dw3x3_pw1x1(64, 128, 2))
        #self.feature.append(dw3x3_pw1x1(128, 128, 1))
        #self.feature.append(dw3x3_pw1x1(128, 256, 2))
        #self.feature.append(dw3x3_pw1x1(256, 256, 1))
        #self.feature.append(dw3x3_pw1x1(256, 512, 2))
        #for i in range(5):
        #    self.feature.append(dw3x3_pw1x1(512, 512, 1))
        #self.feature.append(dw3x3_pw1x1(512, 1024, 2))
        #self.feature.append(dw3x3_pw1x1(1024, 1024, 1))

        self.pool1 = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):

        for i, block in enumerate(self.feature):
            x = block(x)

        x = self.pool1(x)
        x = x.view(-1, 1024)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = MobileNetV1()
    print(model)
