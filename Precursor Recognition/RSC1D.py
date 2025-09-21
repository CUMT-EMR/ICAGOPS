import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ChannelAttention(nn.Module):
    """Temporal Channel Attention (SE Block)"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: (B, C, T)
        m = x.mean(dim=-1)  # (B, C)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(m))))  # (B, C)
        s = s.unsqueeze(-1)  # (B, C, 1)
        return x * s


class FreqGate(nn.Module):
    """Frequency-domain three-part energy cross-gating"""
    def __init__(self, channels, seq_len):
        super().__init__()
        self.fc1 = nn.Linear(3, channels)
        self.fc2 = nn.Linear(channels, channels)
        self.seq_len = seq_len

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        # FFT
        ffted = torch.fft.rfft(x, dim=-1)  # (B, C, T//2+1)
        amp = torch.abs(ffted)

        n_freq = amp.shape[-1]
        split = n_freq // 3
        bands = [
            amp[:, :, :split],          # low
            amp[:, :, split:2*split],   # mid
            amp[:, :, 2*split:]         # high
        ]
        e = torch.stack([b.mean(dim=-1) for b in bands], dim=-1)  # (B, C, 3)
        g = torch.sigmoid(self.fc2(F.relu(self.fc1(e))))  # (B, C, C)
        g = g.mean(dim=-1, keepdim=True) 

        return x * g


class ResidualConvBlock(nn.Module):
    """One-Dimensional Depthwise Attention Residual Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, seq_len=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=(kernel_size//2)*dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding=(kernel_size//2)*dilation, dilation=dilation)
        self.ca = ChannelAttention(out_channels)
        self.fg = FreqGate(out_channels, seq_len)
        self.proj = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # x: (B, C, T)
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = self.ca(y)
        y = self.fg(y)
        return F.relu(y + self.proj(x))

class RSC1D(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=32, num_layers=3, seq_len=1024):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            layers.append(ResidualConvBlock(in_c, hidden_channels, seq_len=seq_len))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)  # (B, hidden_channels, T)



