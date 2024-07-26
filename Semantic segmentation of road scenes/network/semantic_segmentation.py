import logging
import cv2
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from network import resnet_d as Resnet_Deep
from network.nn.mynn import initialize_weights, Norm2d, Upsample
from network.nn.operators import PSPModule,_AtrousSpatialPyramidPoolingModule
from network.dual_branch import dualgcn
from network.Transformer import Transformer
from torch.nn import Softmax
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Center_boundary_splitting(nn.Module):
    def __init__(self, inplane, norm_layer):
        """
        implementation of body generation part
        :param inplane:
        :param norm_layer:
        """
        super(Center_boundary_splitting, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            norm_layer(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )

        self.flow_make = nn.Conv2d(inplane *2 , 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        size = x.size()[2:]
        seg_down = self.down(x)
        seg_down = F.upsample(seg_down, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


class DeepV3Plus(nn.Module):

    def __init__(self, num_classes, trunk='resnet-101', criterion=None, variant='D',
                 skip='m1', skip_num=48):
        super(DeepV3Plus, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num

        if trunk == 'resnet-18':
            resnet = Resnet_Deep.resnet18()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-34':
            resnet = Resnet_Deep.resnet34()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-50':
            resnet = Resnet_Deep.resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101':
            resnet = Resnet_Deep.resnet101()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-152':
            resnet = Resnet_Deep.resnet152()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            print("Not using Dilation ")

        self.aspp = _AtrousSpatialPyramidPoolingModule(2048, 256,
                                                       output_stride=8)

        if self.skip == 'm1':
            self.bot_fine = nn.Conv2d(256, self.skip_num, kernel_size=1, bias=False)
        elif self.skip == 'm2':
            self.bot_fine = nn.Conv2d(512, self.skip_num, kernel_size=1, bias=False)
        else:
            raise Exception('Not a valid skip')

        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.dimensionality = nn.Conv2d(256, 256, kernel_size=2, stride=2, bias=False)#降维
        self.ascension = nn.Conv2d(256, 256, kernel_size=1, bias=False)#升维

        # body_edge module
        self.squeeze_body_edge = Center_boundary_splitting(256, Norm2d)

        # fusion different edge part
        self.edge_fusion = nn.Conv2d(256 + 48, 256, 1, bias=False)
        self.sigmoid_edge = nn.Sigmoid()
        self.edge_out = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=3, padding=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, kernel_size=1, bias=False)
        )

        # DSN for seg body part
        self.dsn_seg_body = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        )

        # Final segmentation part
        self.final_seg = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

    def forward(self, x, gts=None):

        x_size = x.size()
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        fine_size = x1.size()
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        xp = self.aspp(x4)

        aspp = self.bot_aspp(xp)

        seg_body, seg_edge = self.squeeze_body_edge(aspp)

        if self.skip == 'm1':
            # use default low-level feature
            dec0_fine = self.bot_fine(x1)
        else:
            dec0_fine = self.bot_fine(x2)

        seg_body = dualgcn(seg_body)
        seg_edge = self.edge_fusion(torch.cat([Upsample(seg_edge, fine_size[2:]), dec0_fine], dim=1))

        seg_edge = self.dimensionality(seg_edge)

        z1 = []
        """将特征图 x 按宽度 W 进行拆分，拆分为行序列"""
        split_tensors = torch.split(seg_edge, 1, dim=2)
        for x1 in split_tensors:
            y1 = Transformer(x1)
            z1.append(y1)
        # 组合张量
        seg_edge = torch.cat(z1, dim=2)
        n, hidden_size, num_tokens = seg_edge.shape
        h = int(num_tokens ** 0.5)  # 假设特征图的高度和宽度相同
        w = h
        c = hidden_size
        seg_edge = seg_edge.reshape(n, c, h, w)

        z2 = []
        seg_edge = seg_edge.transpose(2, 3)
        """将特征图 x 按高度 H 进行拆分，拆分为列序列"""
        split_tensors = torch.split(seg_edge, 1, dim=2)
        for x2 in split_tensors:
            y2 = Transformer(x2)
            z2.append(y2)
        # 组合张量
        seg_edge = torch.cat(z2, dim=2)
        n, hidden_size, num_tokens = seg_edge.shape
        h = int(num_tokens ** 0.5)  # 假设特征图的高度和宽度相同
        w = h
        c = hidden_size
        seg_edge = seg_edge.reshape(n, c, h, w)
        seg_edge = seg_edge.transpose(2, 3)

        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        seg_edge = upsample(seg_edge)
        seg_edge = self.ascension(seg_edge)

        seg_edge_out = self.edge_out(seg_edge)

        seg_out = seg_edge + Upsample(seg_body, fine_size[2:])
        aspp = Upsample(aspp, fine_size[2:])

        seg_out = torch.cat([aspp, seg_out], dim=1)
        seg_final = self.final_seg(seg_out)

        seg_edge_out = Upsample(seg_edge_out, x_size[2:])
        seg_edge_out = self.sigmoid_edge(seg_edge_out)

        seg_final_out = Upsample(seg_final, x_size[2:])

        seg_body_out = Upsample(self.dsn_seg_body(seg_body), x_size[2:])

        if self.training:
             return self.criterion((seg_final_out, seg_body_out, seg_edge_out), gts)

        return seg_final_out, seg_body_out, seg_edge_out


def Transformer_dual_branch(num_classes, criterion):
    """
    ResNet-101 Based Network
    """
    return DeepV3Plus(num_classes, trunk='resnet-101', criterion=criterion, variant='D', skip='m1')
