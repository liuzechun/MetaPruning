import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

stage_repeat = [3, 4, 6, 3]

channel_scale = []
for i in range(31):
    channel_scale += [(10 + i * 3)/100]

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class first_conv_block(nn.Module):
    def __init__(self, base_inplanes, base_planes, stride):
        super(first_conv_block, self).__init__()

        max_inp = base_inplanes
        max_oup = base_planes

        self.max_inp = max_inp
        self.max_oup = max_oup
        self.stride = stride

        self.fc11 = nn.Linear(1, 32)
        self.fc12 = nn.Linear(32, self.max_oup * self.max_inp * 7 * 7)

        self.first_bn = nn.ModuleList()
        for oup_scale in channel_scale:
            oup = int(self.max_oup * oup_scale)
            self.first_bn.append(nn.BatchNorm2d(oup, affine=False))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x, oup_scale_id):

        oup_scale = channel_scale[oup_scale_id]
        oup = int(self.max_oup * oup_scale)
        scale_tensor = torch.FloatTensor([oup_scale]).to(x.device)

        fc11_out = F.relu(self.fc11(scale_tensor))
        conv1_weight = self.fc12(fc11_out).view(self.max_oup, self.max_inp, 7, 7)

        out = F.conv2d(x, conv1_weight[:oup, :, :, :], bias=None, stride=self.stride, padding=3)
        out = self.first_bn[oup_scale_id](out)
        out = F.relu(out)

        out = self.maxpool(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, base_inplanes, base_planes, stride=1, is_downsample=False):
        super(Bottleneck, self).__init__()
        expansion = 4

        max_inp = base_inplanes
        max_oup = base_planes
        max_mid = int(base_planes/expansion)

        self.max_inp = max_inp
        self.max_oup = max_oup
        self.max_mid = max_mid
        self.stride = stride

        self.fc11 = nn.Linear(3, 32)
        self.fc12 = nn.Linear(32, max_mid * max_inp * 1 * 1)

        self.fc21 = nn.Linear(3, 32)
        self.fc22 = nn.Linear(32, max_mid * max_mid * 3 * 3)

        self.fc31 = nn.Linear(3, 32)
        self.fc32 = nn.Linear(32, max_oup * max_mid * 1 * 1)

        self.bn1 = nn.ModuleList()
        for mid_scale in channel_scale:
            mid = int(self.max_mid * mid_scale)
            self.bn1.append(nn.BatchNorm2d(mid, affine=False))

        self.bn2 = nn.ModuleList()
        for mid_scale in channel_scale:
            mid = int(self.max_mid * mid_scale)
            self.bn2.append(nn.BatchNorm2d(mid, affine=False))

        self.bn3 = nn.ModuleList()
        for oup_scale in channel_scale:
            oup = int(self.max_oup * oup_scale)
            self.bn3.append(nn.BatchNorm2d(oup, affine=False))

        self.is_downsample = is_downsample

        if is_downsample:
            self.fc11_downsample = nn.Linear(3, 32)
            self.fc12_downsample = nn.Linear(32, max_oup * max_inp * 1 * 1)

            self.bn_downsample = nn.ModuleList()
            for oup_scale in channel_scale:
                oup = int(self.max_oup * oup_scale)
                self.bn_downsample.append(nn.BatchNorm2d(oup, affine=False))

    def forward(self, x, mid_scale_id, inp_scale_id, oup_scale_id):

        identity = x

        mid_scale = channel_scale[mid_scale_id]
        inp_scale = channel_scale[inp_scale_id]
        oup_scale = channel_scale[oup_scale_id]

        mid = int(self.max_mid * mid_scale)
        inp = int(self.max_inp * inp_scale)
        oup = int(self.max_oup * oup_scale)

        scale_ratio_tensor = torch.FloatTensor([mid_scale, inp_scale, oup_scale]).to(x.device)

        fc11_out = F.relu(self.fc11(scale_ratio_tensor))
        conv1_weight = self.fc12(fc11_out).view(self.max_mid, self.max_inp, 1, 1)

        fc21_out = F.relu(self.fc21(scale_ratio_tensor))
        conv2_weight = self.fc22(fc21_out).view(self.max_mid, self.max_mid, 3, 3)

        fc31_out = F.relu(self.fc31(scale_ratio_tensor))
        conv3_weight = self.fc32(fc31_out).view(self.max_oup, self.max_mid, 1, 1)

        out = F.conv2d(x, conv1_weight[:mid, :inp, :, :], bias=None, stride=1, padding=0, groups=1)
        out = self.bn1[mid_scale_id](out)
        out = F.relu(out)

        out = F.conv2d(out, conv2_weight[:mid, :mid, :, :], bias=None, stride=self.stride, padding=1, groups=1)
        out = self.bn2[mid_scale_id](out)
        out = F.relu(out)

        out = F.conv2d(out, conv3_weight[:oup, :mid, :, :], bias=None, stride=1, padding=0, groups=1)
        out = self.bn3[oup_scale_id](out)

        if self.is_downsample:
            fc11_downsample_out = F.relu(self.fc11_downsample(scale_ratio_tensor))
            conv1_downsample_weight = self.fc12_downsample(fc11_downsample_out).view(self.max_oup, self.max_inp, 1, 1)

            identity = F.conv2d(x, conv1_downsample_weight[:oup, :inp, :, :], bias=None, stride=self.stride, padding=0, groups=1)

            identity = self.bn_downsample[oup_scale_id](identity)

        out += identity
        out = F.relu(out)

        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.layers = nn.ModuleList()

        #stage0
        self.layers.append(first_conv_block(3, 64, stride=2))

        #stage1
        self.layers.append(Bottleneck(64, 256, stride=1, is_downsample=True))
        for i in range(1, stage_repeat[0]):
            self.layers.append(Bottleneck(256, 256))

        #stage2
        self.layers.append(Bottleneck(256, 512, stride=2, is_downsample=True))
        for i in range(1, stage_repeat[1]):
            self.layers.append(Bottleneck(512, 512))

        #stage3
        self.layers.append(Bottleneck(512, 1024, stride=2, is_downsample=True))
        for i in range(1, stage_repeat[2]):
            self.layers.append(Bottleneck(1024, 1024))

        #stage4
        self.layers.append(Bottleneck(1024, 2048, stride=2, is_downsample=True))
        for i in range(1, stage_repeat[3]):
            self.layers.append(Bottleneck(2048, 2048))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x, stage_oup_scale_ids, mid_scale_ids):

        for i, block in enumerate(self.layers):
            if i == 0:
                x = block(x, stage_oup_scale_ids[i])
            else:
                x = block(x, mid_scale_ids[i-1], stage_oup_scale_ids[i-1], stage_oup_scale_ids[i])

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


