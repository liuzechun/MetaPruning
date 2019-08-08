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

class conv_3x3(nn.Module):
    def __init__(self, base_inp, base_oup, stride):
        super(conv_3x3, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        self.max_scale = channel_scale[-1]
        self.base_oup = base_oup
        self.base_inp = base_inp

        self.max_oup_channel = int(self.max_scale * self.base_oup)
        #self.conv1_weight = nn.Parameter(torch.randn(self.max_oup_channel, base_inp, 3, 3))

        self.fc11 = nn.Linear(1, 32)
        self.fc12 = nn.Linear(32, self.max_oup_channel * self.base_inp * 3 * 3)

        self.first_bn = nn.ModuleList()
        for oup_scale in channel_scale:
            oup = int(self.base_oup * oup_scale)
            self.first_bn.append(nn.BatchNorm2d(oup))

    def forward(self, x, oup_scale_id):

        oup_scale = channel_scale[oup_scale_id]

        oup = int(self.base_oup * oup_scale)

        scale_tensor = torch.FloatTensor([oup_scale/self.max_scale]).to(x.device)

        fc11_out = F.relu(self.fc11(scale_tensor))
        conv1_weight = self.fc12(fc11_out).view(self.max_oup_channel, self.base_inp, 3, 3)

        out = F.conv2d(x, conv1_weight[:oup, :, :, :], bias=None, stride=self.stride, padding=1)
        out = self.first_bn[oup_scale_id](out)
        out = F.relu(out)

        return out

class dw3x3_pw1x1(nn.Module):
    def __init__(self, base_inp, base_oup, stride):
        super(dw3x3_pw1x1, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        self.max_scale = channel_scale[-1]

        self.base_inp = base_inp
        self.base_oup = base_oup

        self.max_inp_channel = int(self.max_scale * self.base_inp)
        self.max_oup_channel = int(self.max_scale * self.base_oup)

        self.fc11 = nn.Linear(2, 32)
        self.fc12 = nn.Linear(32, self.max_inp_channel * 1 * 3 * 3)

        self.fc21 = nn.Linear(2, 32)
        self.fc22 = nn.Linear(32, self.max_oup_channel * self.max_inp_channel * 1 * 1)

        self.first_bn = nn.ModuleList()
        for inp_scale in channel_scale:
            inp = int(self.base_inp * inp_scale)
            self.first_bn.append(nn.BatchNorm2d(inp))

        self.second_bn = nn.ModuleList()
        for oup_scale in channel_scale:
            oup = int(self.base_oup * oup_scale)
            self.second_bn.append(nn.BatchNorm2d(oup, affine=False))


    def forward(self, x, inp_scale_id, oup_scale_id):

        inp_scale = channel_scale[inp_scale_id]
        oup_scale = channel_scale[oup_scale_id]

        inp = int(self.base_inp * inp_scale)
        oup = int(self.base_oup * oup_scale)

        scale_tensor = torch.FloatTensor([inp_scale/self.max_scale, oup_scale/self.max_scale]).to(x.device)

        fc11_out = F.relu(self.fc11(scale_tensor))
        depconv3x3_weight = self.fc12(fc11_out).view(self.max_inp_channel, 1, 3, 3)

        fc21_out = F.relu(self.fc21(scale_tensor))
        conv1x1_weight = self.fc22(fc21_out).view(self.max_oup_channel, self.max_inp_channel, 1, 1)

        out = F.conv2d(x, depconv3x3_weight[:inp, :, :, :], bias=None, stride=self.stride, padding=1, groups=inp)
        out = self.first_bn[inp_scale_id](out)
        out = F.relu(out)

        out = F.conv2d(out, conv1x1_weight[:oup, :inp, :, :], bias=None, stride=1, padding=0, groups=1)
        out = self.second_bn[oup_scale_id](out)
        out = F.relu(out)

        return out

class MobileNetV1(nn.Module):
    def __init__(self, input_size=224, num_classes=1000):
        super(MobileNetV1, self).__init__()

        self.feature = nn.ModuleList()
        self.feature.append(conv_3x3(3, 32, 2))
        self.feature.append(dw3x3_pw1x1(32, 64, 1))
        self.feature.append(dw3x3_pw1x1(64, 128, 2))
        self.feature.append(dw3x3_pw1x1(128, 128, 1))
        self.feature.append(dw3x3_pw1x1(128, 256, 2))
        self.feature.append(dw3x3_pw1x1(256, 256, 1))
        self.feature.append(dw3x3_pw1x1(256, 512, 2))
        for i in range(5):
            self.feature.append(dw3x3_pw1x1(512, 512, 1))
        self.feature.append(dw3x3_pw1x1(512, 1024, 2))
        self.feature.append(dw3x3_pw1x1(1024, 1024, 1))

        self.pool1 = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x, rngs):

        for i, block in enumerate(self.feature):
            if i == 0:
                x = block(x, rngs[0])
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
