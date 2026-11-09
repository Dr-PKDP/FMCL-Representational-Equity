"""
models.py

Compact networks. Compactness is a design requirement of the study (device-deployable
models): the architectures are fixed and are not enlarged.

    CompactECGNet   input (B, 12, 1000)    164,289 parameters
    CompactDermNet  input (B, 3, 128, 128) 302,425 parameters
Both output raw logits of shape (B, 1); apply a sigmoid in the loss / evaluation.
"""
import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CompactECGNet(nn.Module):
    def __init__(self, n_leads=12, n_classes=1, base_ch=32):
        super().__init__()
        self.stem = ConvBlock1D(n_leads, base_ch, kernel_size=15, stride=2)
        self.block1 = ConvBlock1D(base_ch, base_ch * 2, kernel_size=9, stride=2)
        self.block2 = ConvBlock1D(base_ch * 2, base_ch * 4, kernel_size=7, stride=2)
        self.block3 = ConvBlock1D(base_ch * 4, base_ch * 4, kernel_size=5, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(base_ch * 4, n_classes)

    def forward(self, x):
        x = self.block3(self.block2(self.block1(self.stem(x))))
        return self.head(self.pool(x).squeeze(-1))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CompactDermNet(nn.Module):
    def __init__(self, n_classes=1, base_ch=24):
        super().__init__()
        self.stem = ConvBlock2D(3, base_ch)
        self.block1 = ConvBlock2D(base_ch, base_ch * 2)
        self.block2 = ConvBlock2D(base_ch * 2, base_ch * 4)
        self.block3 = ConvBlock2D(base_ch * 4, base_ch * 4)
        self.block4 = ConvBlock2D(base_ch * 4, base_ch * 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(base_ch * 8, n_classes)

    def forward(self, x):
        x = self.block4(self.block3(self.block2(self.block1(self.stem(x)))))
        return self.head(self.pool(x).flatten(1))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    e, d = CompactECGNet(), CompactDermNet()
    assert e(torch.randn(8, 12, 1000)).shape == (8, 1) and d(torch.randn(8, 3, 128, 128)).shape == (8, 1)
    print(f"CompactECGNet params={e.n_params():,}  CompactDermNet params={d.n_params():,}  forward OK")
    assert e.n_params() == 164289 and d.n_params() == 302425
