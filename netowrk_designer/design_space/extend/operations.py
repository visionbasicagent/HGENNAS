import torch
import math
import torch.nn as nn
from torch.autograd import Variable


OPS = {
    'none'         : lambda C_in, C_out, kernel_size, stride, affine, se : Zero(C_in, C_out, stride),
    'skip_connect': lambda C_in, C_out, kernel_size, stride, affine, se: Identity() if stride == 1 else FactorizedReduce(C_in, C_out, affine=affine),
    'avg_pool': lambda  C_in, C_out, kernel_size, stride, affine, se: POOLING(C_in, C_out, kernel_size, stride=stride, affine=affine, mode='avg'),
    'conv': lambda  C_in, C_out, kernel_size, stride, affine, se: ReLUConvBN( C_in, C_out, kernel_size=kernel_size, stride=stride, affine=affine, se=se),
    'sep_conv': lambda  C_in, C_out, kernel_size, stride, affine, se: SepConv( C_in, C_out, kernel_size=kernel_size, stride=stride, affine=affine, se=se),
    'dil_conv': lambda  C_in, C_out, kernel_size, stride, affine, se : DilConv( C_in, C_out, kernel_size=kernel_size, stride=stride, affine=affine, se=se),
    'mbonv_2': lambda  C_in, C_out, kernel_size, stride, affine, se: MBConv( C_in, C_out, kernel_size=kernel_size, stride=stride, exp=2, affine=affine, se=se),
    'mbonv_3': lambda  C_in, C_out, kernel_size, stride, affine, se: MBConv( C_in, C_out, kernel_size=kernel_size, stride=stride, exp=3, affine=affine, se=se),
    'mbonv_6': lambda  C_in, C_out, kernel_size, stride, affine, se: MBConv( C_in, C_out, kernel_size=kernel_size, stride=stride, exp=6, affine=affine,se=se),
    'bconv_05': lambda  C_in, C_out, kernel_size, stride, affine, se: MBConv( C_in, C_out, kernel_size=kernel_size, stride=stride, exp=0.5, affine=affine, se=se),
    'bconv_025': lambda  C_in, C_out, kernel_size, stride, affine, se: MBConv( C_in, C_out, kernel_size=kernel_size, stride=stride, exp=0.25, affine=affine,se=se),
}


NF_GRAPH_SEARCH_FULL = ['skip_connect', 'avg_pool', 'conv', 'sep_conv', 'dil_conv', 'mbonv_2', 'mbonv_3', 'mbonv_6', 'bconv_05', 'bconv_025']

SearchSpaceNames = {'nf_graph' : NF_GRAPH_SEARCH_FULL}

class Zero(nn.Module):

    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in   = C_in
        self.C_out  = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1: return x.mul(0.)
            else               : return x[:,:,::self.stride,::self.stride].mul(0.)
        else: ## this is never called in nasbench201
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            zeros = zeros[:,:,::self.stride,::self.stride]
            return zeros

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, affine=True, se=False):
        super(ReLUConvBN, self).__init__()
        padding = math.floor(kernel_size/2)
        if se :
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
            )
        else:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(C_out, affine=affine)
            )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, affine=True, se=False):
        super(DilConv, self).__init__()
        dilation = 2
        padding =  int((dilation * (kernel_size - 1)) / 2)
        if se:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                        groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
            )
        
        else:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                        groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
            )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, affine=True, se=False):
        super(SepConv, self).__init__()
        padding = math.floor(kernel_size/2)
        if se:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
            )
        else:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
            )


    def forward(self, x):
        return self.op(x)

class MBConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, exp, affine=True, se=False):
        super(MBConv, self).__init__()
        padding = math.floor(kernel_size/2)
        C_exp = int(C_in * exp)
        if exp == 1:
            if se:
                self.op = nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
                    nn.BatchNorm2d(C_in, affine=affine),
                    nn.ReLU(inplace=False),
                    SE(C_in),
                    nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(C_out, affine=affine),
                )
            else:
                self.op = nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
                    nn.BatchNorm2d(C_in, affine=affine),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(C_out, affine=affine),
                )

        else:
            if se:
                self.op = nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in, C_exp, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(C_exp, affine=affine),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_exp, C_exp, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_exp, bias=False),
                    nn.BatchNorm2d(C_exp, affine=affine),
                    nn.ReLU(inplace=False),
                    SE(C_exp),
                    nn.Conv2d(C_exp, C_out, 1, 1 , 0, bias=False),
                    nn.BatchNorm2d(C_out, affine=affine),
                )
            else:
                self.op = nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in, C_exp, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(C_exp, affine=affine),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_exp, C_exp, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_exp, bias=False),
                    nn.BatchNorm2d(C_exp, affine=affine),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_exp, C_out, 1, 1 , 0, bias=False),
                    nn.BatchNorm2d(C_out, affine=affine),
                )
            
    def forward(self, x):
        return self.op(x)
        
        
        
class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        #print(out, 'FactorizedReduce')
        return out

class POOLING(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, mode, affine=True, track_running_stats=True):
    super(POOLING, self).__init__()
    padding = math.floor(kernel_size/2)
    if mode == 'avg'  : 
        self.op = nn.Sequential(
            nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=False), 
            nn.BatchNorm2d(C_in, affine=False)
        )
    elif mode == 'max': 
        self.op = nn.Sequential(
            nn.MaxPool2d(kernel_size, stride=stride, padding=padding), 
            nn.BatchNorm2d(C_in, affine=False)
        )
        
    if C_in != C_out:
        self.out = nn.Sequential(
                                nn.ReLU(inplace=False),
                                nn.Conv2d(C_in, C_out, 1, 1),
                                nn.BatchNorm2d(C_out),
                                )
    else:
        self.out = None

  def forward(self, inputs, block_input=False):
    if self.out:
        return self.out(self.op(inputs))
    else:
        return self.op(inputs)


class ResNetBasicblock(nn.Module):
    def __init__(self, inplanes, planes, stride, affine=True):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, affine)
        self.conv_b = ReLUConvBN(  planes, planes, 3,      1, affine)
        if stride == 2:
            self.downsample = nn.Sequential(
                                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, affine)
        else:
            self.downsample = None
        self.in_dim  = inplanes
        self.out_dim = planes
        self.stride  = stride
        self.num_conv = 2

    def extra_repr(self):
        string = '{name}(inC={in_dim}, outC={out_dim}, stride={stride})'.format(name=self.__class__.__name__, **self.__dict__)
        return string

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock

def sigmoid(x, inplace: bool = False):
    return x.sigmoid_() if inplace else x.sigmoid()


def make_divisible(v: int, divisor: int = 8, min_value: int = None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:  # ensure round down does not go down by more than 10%.
        new_v += divisor
    return new_v

class SE(nn.Module):

    def __init__(self, out_channels=None):
        super(SE, self).__init__()
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.se_ratio = 0.25
        self.se_channels = max(1, int(round(self.out_channels * self.se_ratio)))

        self.netblock = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.se_channels, kernel_size=1, stride=1,
                        padding=0, bias=False),
            nn.BatchNorm2d(self.se_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.se_channels, out_channels=self.out_channels, kernel_size=1, stride=1,
                        padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_x = self.netblock(x)
        return se_x * x
