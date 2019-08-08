import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from resnet import channel_scale

stage_repeat = [3, 4, 6, 3]
stage_out_channel = [64] + [256] * 3 + [512] * 4 + [1024] * 6 + [2048] * 3

def adapt_channel(ids):

    stage_oup_scale_ids = []
    stage_oup_scale_ids += [ids[0]]
    for i in range(len(stage_repeat)-1):
        stage_oup_scale_ids += [ids[i+1]] * stage_repeat[i]
    stage_oup_scale_ids +=[-1] * stage_repeat[-1]

    mid_scale_ids = ids[len(stage_repeat):]

    overall_channel = []
    mid_channel = []
    for i in range(len(stage_out_channel)):
        if i == 0 :
            overall_channel += [int(stage_out_channel[i] * channel_scale[stage_oup_scale_ids[i]])]
        else:
            overall_channel += [int(stage_out_channel[i] * channel_scale[stage_oup_scale_ids[i]])]
            mid_channel += [int(stage_out_channel[i]//4 * channel_scale[mid_scale_ids[i-1]])]

    return overall_channel, mid_channel


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    def __init__(self, midplanes, inplanes, planes, stride=1, is_downsample=False):
        super(Bottleneck, self).__init__()
        expansion = 4

        #midplanes = int(planes/expansion)
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, midplanes)
        self.bn1 = norm_layer(midplanes)
        self.conv2 = conv3x3(midplanes, midplanes, stride)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = conv1x1(midplanes, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes
        self.midplanes = midplanes

        self.is_downsample = is_downsample
        self.expansion = expansion

        if is_downsample:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride=stride),
                norm_layer(planes),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.is_downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class resnet50(nn.Module):
    def __init__(self, ids, num_classes=1000):
        super(resnet50, self).__init__()

        overall_channel, mid_channel = adapt_channel(ids)

        layer_num =0
        self.conv1 = nn.Conv2d(3, overall_channel[layer_num], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(overall_channel[layer_num])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList()

        layer_num += 1
        for i in range(len(stage_repeat)):
            if i == 0:
                self.layers.append(Bottleneck(mid_channel[layer_num-1], overall_channel[layer_num-1], overall_channel[layer_num], stride=1, is_downsample=True))
                layer_num += 1
            else:
                self.layers.append(Bottleneck(mid_channel[layer_num-1], overall_channel[layer_num-1], overall_channel[layer_num], stride=2, is_downsample=True))
                layer_num += 1

            for j in range(1, stage_repeat[i]):
                self.layers.append(Bottleneck(mid_channel[layer_num-1], overall_channel[layer_num-1], overall_channel[layer_num]))
                layer_num +=1

        ##stage1
        #self.layers.append(Bottleneck(64, 256, stride=1, is_downsample=True))
        #for i in range(1, stage_repeat[0]):
        #    self.layers.append(Bottleneck(256, 256))

        ##stage2
        #self.layers.append(Bottleneck(256, 512, stride=2, is_downsample=True))
        #for i in range(1, stage_repeat[1]):
        #    self.layers.append(Bottleneck(512, 512))

        ##stage3
        #self.layers.append(Bottleneck(512, 1024, stride=2, is_downsample=True))
        #for i in range(1, stage_repeat[2]):
        #    self.layers.append(Bottleneck(1024, 1024))

        ##stage4
        #self.layers.append(Bottleneck(1024, 2048, stride=2, is_downsample=True))
        #for i in range(1, stage_repeat[3]):
        #    self.layers.append(Bottleneck(2048, 2048))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, block in enumerate(self.layers):
            x = block(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


