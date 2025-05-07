import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
# GitHub地址 ：https://github.com/apple1986/HWD
# 论文地址：https://www.sciencedirect.com/science/article/pii/S0031320323005174
# 论文：Haar wavelet downsampling: A simple but effective downsampling module

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            # nn.LayerNorm(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

class UNetDown(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UNetDown, self).__init__()
        block = []

        block.append(Down_wt(in_chans, out_chans))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


if __name__ == '__main__':
    block = UNetDown(32, 64)  # 输入通道数，输出通道数
    input = torch.rand(3, 32, 64, 64)  # 输入tensor形状B C H W
    output = block(input)
    print(output.size())
