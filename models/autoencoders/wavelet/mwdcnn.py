
'''
    This model is dynamic conv + wavelet transform + Residual dense block callde DWD
'''


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from pytorch_wavelets import DWTForward, DWTInverse   # (or import DWT, IDWT)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


def dynamic_conv(in_channels, out_channels, kernel_size, k=4, bias=True):
    return dynamic_conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, K=k
    )

class attention2d(nn.Module):
    def __init__(self, in_planes, K, ):
        super(attention2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, K, 1,)
        self.fc2 = nn.Conv2d(K, K, 1,)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x, 1)


class dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding=0, bias=True, stride=1, dilation=1, groups=1, K=4,):
        super(dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, K, )

        self.weight = nn.Parameter(torch.Tensor(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
            self._init_weight(self.bias)
        else:
            self.bias = None

        # 初始化权重参数
        self._init_weight(self.weight)

    def forward(self, x):   # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)    # 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)

        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

    def _init_weight(self, weight):
        nn.init.xavier_uniform_(weight)

class WRB(nn.Module):
    def __init__(self, growth_rate, num_layers, in_channels, wave='db3'):
        super(WRB, self).__init__()

        n_feats = in_channels*4
        growth_rate = growth_rate

        self.RDB = RDB(n_feats, growth_rate, num_layers)

        # self.conv1 = nn.Conv2d(in_channels*4, in_channels*2, kernel_size=1)     #
        self.conv2 = nn.Conv2d(n_feats+num_layers*growth_rate, in_channels*4, kernel_size=1)

        # J为分解的层次数,wave表示使用的变换方法
        self.WTF = DWTForward(J=1, mode='zero', wave=wave)  # Accepts all wave types available to PyWavelets
        self.WTI = DWTInverse(mode='zero', wave=wave)

    def forward(self, x):
        batch_size, _, h, w = x.shape

        if h % 2 == 1 and w % 2 == 1:
            pad = nn.ReplicationPad2d(1)
            x = pad(x)
            x = x[:, :, 0:-1, 0:-1]
        elif h % 2 == 1:
            pad = nn.ReplicationPad2d(1)
            x = pad(x)
            x = x[:, :, 0:-1, 1:-1]
        elif w % 2 == 1:
            pad = nn.ReplicationPad2d(1)
            x = pad(x)
            x = x[:, :, 1:-1, 0:-1]

        yl, yh = self.WTF(x)
        yh = yh[0]  # 返回的yh是一个list
        fh, fw = yh.shape[-2], yh.shape[-1]
        yh = yh.view(batch_size, -1, fh, fw)

        out = torch.cat((yl, yh), 1)

        out2 = self.RDB(out, False)



        out3 = self.conv2(out2)



        out3 = out3+out

        yl = out3[:, 0:(yl.shape[1]), :, :]
        yh = out3[:, yl.shape[1]:, :, :].view(batch_size, -1, 3, fh, fw)
        yh = [yh, ]
        out = self.WTI((yl, yh))

        if h % 2 == 1 and w % 2 == 1:
            out = out[:, :, 1:, 1:]
        elif h % 2 == 1:
            out = out[:, :, 1:, :]
        elif w % 2 == 1:
            out = out[:, :, :, 1:]

        return out

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size//2), bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, kernel_size=5):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate, kernel_size=kernel_size) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)  #k=1

    def forward(self, x, lrl:bool=True):
        if lrl:
            return x + self.lff(self.layers(x))  # local residual learning
        else:
            return self.layers(x)


class MeanShift(nn.Conv2d):
    def __init__(
                self, rgb_range,
                rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1
                ):

        super(MeanShift, self).__init__(3, 3, kernel_size=(1, 1))
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
                self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
                bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
                self, conv, n_feats, kernel_size,
                bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def make_model(args):
    return WMDCNN(args), 1

class WMDCNN(nn.Module):
    def __init__(self, n_colors, n_feats, growth_rate, RDB_num_layers, conv=default_conv):
        super(WMDCNN, self).__init__()

        kernel_size = 5
        growth_rate = growth_rate
        rdb_num_layers = RDB_num_layers

        self.conv1 = conv(n_colors, n_feats, kernel_size)  # conv1

        self.dy_conv_block = nn.Sequential(
            dynamic_conv(n_feats, n_feats, kernel_size),
            nn.ReLU(True)
        )

        self.conv_block1 = nn.Sequential(
            conv(n_feats, n_feats, kernel_size),
            nn.ReLU(True)
        )

        self.WRB1 = WRB(growth_rate, RDB_num_layers, n_feats)
        self.WRB2 = WRB(growth_rate, rdb_num_layers, n_feats)

        self.RDB_1 = nn.Sequential(
            RDB(n_feats, growth_rate, rdb_num_layers),
            nn.ReLU(True)
        )

        self.RDB_2 = nn.Sequential(
            RDB(n_feats, growth_rate, rdb_num_layers),
            nn.ReLU(True)
        )

        self.conv_block2 = nn.Sequential(
            conv(n_feats, n_feats, kernel_size),
            nn.ReLU(True)
        )

        self.conv2 = conv(n_feats, n_colors, kernel_size)  # conv1

        self.seq = nn.Sequential(
            self.conv1,
            self.dy_conv_block,
            self.conv_block1,
            self.WRB1,
            self.WRB2
        )

    def forward(self, x):
        y = x

        out1 = self.seq(x)

        out2 = self.RDB_1(out1)
        out3 = self.RDB_2(out2)

        out4 = out1 + out2 + out3

        out5 = self.conv_block2(out4)
        out = self.conv2(out5)

        return y - out


if __name__ == '__main__':
    device = torch.device("cpu:0")
    model = WMDCNN(2, 4, 4, 28).to(device).eval()
    x = torch.randn(16, 2, 80, 80, device=device)
    print(model(x).size())
    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
