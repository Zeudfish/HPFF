import copy
import pdb
import time

import torch
import torch.nn as nn
import math

from .config import InfoPro, InfoPro_balanced_memory
from .auxiliary_net import Decoder, AuxClassifier


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class InfoProResNet(nn.Module):

    def __init__(self, block, layers, arch, local_module_num, batch_size, image_size=32,
                 balanced_memory=False, dataset='cifar10', class_num=10,
                 wide_list=(16, 16, 32, 64), dropout_rate=0,
                 aux_net_config='0c2f', local_loss_mode='contrast',
                 aux_net_widen=1, aux_net_feature_dim=128,wx=0.001,wy=0.999):
        super(InfoProResNet, self).__init__()

        assert arch in ['resnet32', 'resnet110'], "This repo supports resnet32 and resnet110 currently. " \
                                                  "For other networks, please set network configs in .configs."
        self.widelist = wide_list
        self.inplanes = wide_list[0]
        self.dropout_rate = dropout_rate
        self.feature_num = wide_list[-1]
        self.class_num = class_num
        self.local_module_num = local_module_num
        self.layers = layers
        self.widen = aux_net_widen
        self.fea = aux_net_feature_dim

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, wide_list[1], layers[0])
        self.layer2 = self._make_layer(block, wide_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, wide_list[3], layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feature_num, self.class_num)
        self.fc64 = nn.Linear(self.widelist[3], self.class_num)
        self.Flatten = nn.Flatten()
        self.wx = wx
        self.wy = wy

        self.criterion_ce = nn.CrossEntropyLoss()

        try:
            self.infopro_config = InfoPro_balanced_memory[arch][dataset][local_module_num] \
                if balanced_memory else InfoPro[arch][local_module_num]
        except:
            raise NotImplementedError


        for item in self.infopro_config:
            module_index, layer_index = item
            exec('self.net_decoder_' + str(module_index) + '_' + str(layer_index) +
                 '= Decoder(wide_list[module_index], image_size, widen=aux_net_widen)')
            exec('self.net_classifier_' + str(module_index) + '_' + str(layer_index)  +
                 '= AuxClassifier(wide_list[module_index], net_config=aux_net_config, '
                 'loss_mode=local_loss_mode, class_num=class_num, '
                 'widen=aux_net_widen, feature_dim=aux_net_feature_dim)')
            exec('self.aux_decoder_' + str(module_index) + '_' + str(layer_index) +
                 '= Decoder(wide_list[module_index], image_size, widen=aux_net_widen)')
            exec('self.aux_classifier_' + str(module_index) + '_' + str(layer_index) +
                 '= AuxClassifier(wide_list[module_index], net_config=aux_net_config, '
                 'loss_mode=local_loss_mode, class_num=class_num, '
                 'widen=aux_net_widen, feature_dim=aux_net_feature_dim)')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if 'cifar' in dataset:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [125.3, 123.0, 113.9]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).cuda()
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [63.0, 62.1, 66.7]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).cuda()
        else:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).cuda()
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).cuda()

        self.Encoder_Net = self._make_Encoder_Net()
        for net1 in self.Encoder_Net:
            net1 = net1.cuda()



    def _image_restore(self, normalized_image):
        return normalized_image.mul(self.mask_train_std[:normalized_image.size(0)]) \
               + self.mask_train_mean[:normalized_image.size(0)]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate=self.dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def _make_Encoder_Net(self):
        Encoder_Net = nn.ModuleList([])

        Encoder_temp = nn.ModuleList([])

        local_block_index = 0

        # Build Encoder_Net
        for blocks in range(len(self.layers)):
            for layers in range(self.layers[blocks]):
                Encoder_temp.append(eval('self.layer' + str(blocks + 1))[layers])
                if blocks + 1 == self.infopro_config[local_block_index][0] \
                        and layers == self.infopro_config[local_block_index][1]:
                    Encoder_Net.append(nn.Sequential(*Encoder_temp))

                    Encoder_temp = nn.ModuleList([])
                    local_block_index += 1
        return Encoder_Net


    def forward_original(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc64(x)

    def forward(self, img, target=None,ixx_1=0, ixy_1=0,
                ixx_2=0, ixy_2=0):
        if self.training:
            # local_module_num = 1 means the E2E training
            if self.local_module_num == 1:
                x = self.conv1(img)
                x = self.bn1(x)
                x = self.relu(x)

                for i in range(len(self.Encoder_Net)):
                    x = self.Encoder_Net[i](x)

                x = self.avgpool(x)
                x = x.view(x.size(0),-1)

                logits = self.fc64(x)
                loss = self.criterion_ce(logits,target)
                loss.backward()

                return logits,logits,loss

            else:
                x = self.conv1(img)
                x = self.bn1(x)
                x = self.relu(x)

                y = torch.clone(x)
                cur = 0


                for i in range(len(self.Encoder_Net) - 1):
                    if i == 0:
                        x = self.Encoder_Net[i](x)
                        x = x.detach()

                    else:
                        x = self.Encoder_Net[i](x)

                        y = self.Encoder_Net[i - 1](y)
                        z = self.Encoder_Net[i](y)
                        ratio = i / (self.local_module_num - 2) if self.local_module_num > 2 else 0
                        ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio
                        ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio
                        local_index, layer_index = self.infopro_config[i]
                        local_x,layer_x = self.infopro_config[i]
                        loss_x = eval('self.net_classifier_' + str(local_index) + '_' + str(layer_index))(x, target)
                        loss_y1 = eval('self.aux_decoder_' + str(local_x) + '_' + str(layer_x))(z,self._image_restore(img))
                        loss_y2 = eval('self.aux_classifier_' + str(local_x) + '_' + str(layer_x))(z,target)
                        loss_y = ixx_r * loss_y1 + ixy_r * loss_y2
                        loss = self.wx * loss_x + self.wy * loss_y
                        loss.backward()

                        x = x.detach()
                        y = y.detach()

                # last local module
                x = self.Encoder_Net[-1](x)
                y = self.Encoder_Net[-2](y)
                y = self.Encoder_Net[-1](y)
                x = self.avgpool(x)
                y = self.avgpool(y)

                x = x.view(x.size(0), -1)
                y = y.view(y.size(0),-1)

                logits_x = self.fc64(x)
                logits_y = self.fc64(y)

                loss_x = self.criterion_ce(logits_x, target)
                loss_y = self.criterion_ce(logits_y, target)
                loss = self.wx * loss_x + self.wy * loss_y

                loss.backward()

            return logits_x,logits_y, loss

        else:
            x = self.conv1(img)
            x = self.bn1(x)
            x = self.relu(x)

            for i in range(len(self.Encoder_Net)):
                x = self.Encoder_Net[i](x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            logits = self.fc64(x)
            loss = self.criterion_ce(logits, target)
            return logits, loss


def resnet20(**kwargs):
    model = InfoProResNet(BasicBlock, [3, 3, 3], arch='resnet20', **kwargs)
    return model


def resnet32(**kwargs):
    model = InfoProResNet(BasicBlock, [5, 5, 5], arch='resnet32', **kwargs)
    return model


def resnet44(**kwargs):
    model = InfoProResNet(BasicBlock, [7, 7, 7], arch='resnet44', **kwargs)
    return model


def resnet56(**kwargs):
    model = InfoProResNet(BasicBlock, [9, 9, 9], arch='resnet56', **kwargs)
    return model


def resnet110(**kwargs):
    model = InfoProResNet(BasicBlock, [18, 18, 18], arch='resnet110', **kwargs)
    return model


def resnet1202(**kwargs):
    model = InfoProResNet(BasicBlock, [200, 200, 200], arch='resnet1202', **kwargs)
    return model


def resnet164(**kwargs):
    model = InfoProResNet(Bottleneck, [18, 18, 18], arch='resnet164', **kwargs)
    return model


def resnet1001(**kwargs):
    model = InfoProResNet(Bottleneck, [111, 111, 111], arch='resnet1001', **kwargs)
    return model

# if __name__ == "__main__":
#     net = resnet110(local_module_num=55, batch_size=256, image_size=32,
#                    balanced_memory=False, dataset='cifar10', class_num=10,
#                    wide_list=(16, 16, 32, 64), dropout_rate=0,
#                    aux_net_config='0c2f', local_loss_mode='contrast',
#                    aux_net_widen=1, aux_net_feature_dim=128)
#     net = net.cuda()
#     x = torch.ones(4,3,32,32).cuda()
#     target = torch.zeros(4).long().cuda()
#     print(net(x, target))
