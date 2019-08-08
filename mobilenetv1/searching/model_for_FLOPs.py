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

class conv_bn(nn.Module):
    def __init__(self, inp, base_oup, oup_scale, stride):
        super(conv_bn, self).__init__()

        assert stride in [1, 2]

        oup = int(base_oup * oup_scale)

        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        return out

class dw3x3_pw1x1(nn.Module):
    def __init__(self, base_inp, base_oup, inp_scale, oup_scale, stride):
        super(dw3x3_pw1x1, self).__init__()

        assert stride in [1, 2]

        inp = int(base_inp * inp_scale)
        oup = int(base_oup * oup_scale)

        self.depconv3x3 = nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.conv1x1 = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)

    def forward(self, x):

        out = self.depconv3x3(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv1x1(out)
        out = self.bn2(out)
        out = F.relu(out)

        return out

class Select_conv1(nn.Module):
    def __init__(self, inp, oup, stride):
        super(Select_conv1, self).__init__()
        self._ops = nn.ModuleList()
        for oup_scale in channel_scale:
            op = conv_bn(inp, oup, oup_scale, stride)
            self._ops.append(op)

    def forward(self, x, inp_id, oup_id):
        oup_id = (oup_id + len(channel_scale)) % len(channel_scale)
        return self._ops[oup_id](x)

class Select_one_OP(nn.Module):
    def __init__(self, inp, oup, stride):
        super(Select_one_OP, self).__init__()
        self._ops = nn.ModuleList()
        for inp_scale in channel_scale:
            for oup_scale in channel_scale:
                op = dw3x3_pw1x1(inp, oup, inp_scale, oup_scale, stride)
                self._ops.append(op)

    def forward(self, x, inp_id, oup_id):
        inp_id = (inp_id + len(channel_scale)) % len(channel_scale)
        oup_id = (oup_id + len(channel_scale)) % len(channel_scale)
        id = inp_id * len(channel_scale) + oup_id
        return self._ops[id](x)

class MobileNetV1(nn.Module):
    def __init__(self, input_size=224, num_classes=1000):
        super(MobileNetV1, self).__init__()

        self.feature = nn.ModuleList()
        self.feature.append(Select_conv1(3, stage_out_channel[0], 2))
        for i in range(1, len(stage_out_channel)):
            if stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != 64:
                self.feature.append(Select_one_OP(stage_out_channel[i-1], stage_out_channel[i], 2))
            else:
                self.feature.append(Select_one_OP(stage_out_channel[i-1], stage_out_channel[i], 1))
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

    def forward(self, x, rngs):

        for i, block in enumerate(self.feature):
            if i == 0:
                x = block(x, -1, rngs[0])
            elif i == 13:
                x = block(x, rngs[i-1], -1)
            else:
                x = block(x, rngs[i-1], rngs[i])

        x = self.pool1(x)
        x = x.view(-1, 1024)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = MobileNetV1()
    print(model)
