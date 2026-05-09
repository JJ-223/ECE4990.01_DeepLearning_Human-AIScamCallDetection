
import torch
import torch.nn as nn

class FMS(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc  = nn.Linear(channels, channels)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        s = x.mean(dim=[2, 3])
        s = self.sig(self.fc(s))
        s = s.unsqueeze(-1).unsqueeze(-1)
        return x * s + s

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, first_block=False):
        super().__init__()
        self.first_block = first_block
        if not first_block:
            self.bn0  = nn.BatchNorm2d(in_ch)
            self.act0 = nn.LeakyReLU(0.3)
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.act1  = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act2  = nn.LeakyReLU(0.3)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None
    def forward(self, x):
        identity = x
        if not self.first_block:
            x = self.act0(self.bn0(x))
        out = self.conv1(self.act1(self.bn1(x)))
        out = self.conv2(self.act2(self.bn2(out)))
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        return out + identity

class SpecRNet(nn.Module):
    """
    SpecRNet audio deepfake detector.
    Input : (B, 1, 80, T) LFCC tensor
    Output: (B, 1) probability — 1=fake, 0=real
    """
    def __init__(self):
        super().__init__()
        self.bn_input  = nn.BatchNorm2d(1)
        self.act_input = nn.SELU()
        self.res1   = ResBlock(1,  32, first_block=True)
        self.pool1a = nn.MaxPool2d(2)
        self.fms1   = FMS(32)
        self.pool1b = nn.MaxPool2d(2)
        self.res2   = ResBlock(32, 64)
        self.pool2a = nn.MaxPool2d(2)
        self.fms2   = FMS(64)
        self.pool2b = nn.MaxPool2d(2)
        self.res3   = ResBlock(64, 64)
        self.pool3a = nn.MaxPool2d(2)
        self.fms3   = FMS(64)
        self.pool3b = nn.MaxPool2d(2)
        self.gru1    = nn.GRU(64,  64, batch_first=True, bidirectional=True)
        self.gru2    = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.fc1     = nn.Linear(128, 64)
        self.act_fc  = nn.ReLU()
        self.fc2     = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.act_input(self.bn_input(x))
        x = self.pool1b(self.fms1(self.pool1a(self.res1(x))))
        x = self.pool2b(self.fms2(self.pool2a(self.res2(x))))
        x = self.pool3b(self.fms3(self.pool3a(self.res3(x))))
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, W, C * H)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = x[:, -1, :]
        return self.sigmoid(self.fc2(self.act_fc(self.fc1(x))))
