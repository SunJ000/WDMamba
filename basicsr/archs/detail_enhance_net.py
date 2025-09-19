import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from basicsr.archs.Ublock import unetBlock

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)


class fft_bench(nn.Module):
    def __init__(self, n_feat):
        super(fft_bench, self).__init__()
        self.main = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1)
        self.mag = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0)
        self.pha = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
        )

        self.eca = ECAAttention()

    def forward(self, x):
        _, _, H, W = x.shape
        fre = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fre)
        pha = torch.angle(fre)

        pha_eca = self.eca(pha)

        mag_out = self.mag(mag)
        pooling = torch.nn.functional.adaptive_avg_pool2d(mag_out, (1, 1))
        pooling = torch.nn.functional.softmax(pooling, dim=1)
        pha1 = pha_eca * pooling
        pha1 = self.pha(pha1)
        pha_out = pha1 + pha
        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        fre_out = torch.complex(real, imag)
        y = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')

        return self.main(x) + y


class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Block, self).__init__()

        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.ublock = unetBlock(dim=dim)

    def forward(self, x):

        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.ublock(res)
        res += x

        return res


class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


class DENet(nn.Module):
    def __init__(self, gps, blocks, conv1=default_conv):
        super(DENet, self).__init__()
        self.gps = gps
        self.dim = 16
        kernel_size = 3

        self.fft1 = fft_bench(self.dim)
        self.fft2 = fft_bench(self.dim)
        self.fft3 = fft_bench(self.dim)

        pre_process = [conv1(3, self.dim, kernel_size)]
        # assert self.gps == 3

        self.g1 = Group(conv1, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv1, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv1, self.dim, kernel_size, blocks=blocks)

        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 4, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 4, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])

        post_precess = [
            conv1(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):

        x = self.pre(x1)

        res1 = self.g1(x)
        res1 = self.fft1(res1)

        res2 = self.g2(res1)
        res2 = self.fft2(res2)

        res3 = self.g3(res2)
        res3 = self.fft3(res3)

        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        x = self.post(out)
        return x + x1


if __name__ == "__main__":
    x = torch.randn(1, 3, 64, 64)
    net = DENet(3,3)
    out = net(x)
    print(out.size())