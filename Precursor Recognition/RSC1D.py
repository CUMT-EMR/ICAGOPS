import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.features)


def conv1d_same(in_channels, out_channels, kernel_size, stride=1, groups=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, groups=groups)


def pad_same(x, kernel_size, stride):
    in_len = x.shape[-1]
    out_len = (in_len + stride - 1) // stride
    p = max(0, (out_len - 1) * stride + kernel_size - in_len)
    pad_left = p // 2
    pad_right = p - pad_left
    return F.pad(x, (pad_left, pad_right), "constant", 0)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=1, groups=1,
                 downsample=False, use_bn=True, use_dropout=True, first=False):
        super().__init__()
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.downsample = downsample
        self.first = first
        self.stride = stride if downsample else 1
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.kernel_size = k_size
        self.groups = groups

        self.conv1 = conv1d_same(in_ch, out_ch, k_size, self.stride, groups)
        self.conv2 = conv1d_same(out_ch, out_ch, k_size, 1, groups)

        if use_bn:
            self.bn1 = nn.BatchNorm1d(in_ch)
            self.bn2 = nn.BatchNorm1d(out_ch)
        if use_dropout:
            self.drop1 = nn.Dropout(0.5)
            self.drop2 = nn.Dropout(0.5)

    def forward(self, x):
        identity = x

        out = x
        if not self.first:
            if self.use_bn:
                out = self.bn1(out)
            out = F.relu(out)
            if self.use_dropout:
                out = self.drop1(out)

        out = pad_same(out, self.kernel_size, self.stride)
        out = self.conv1(out)

        if self.use_bn:
            out = self.bn2(out)
        out = F.relu(out)
        if self.use_dropout:
            out = self.drop2(out)

        out = pad_same(out, self.kernel_size, 1)
        out = self.conv2(out)

        # Downsample identity if needed
        if self.downsample:
            identity = pad_same(identity, self.stride, self.stride)
            identity = F.max_pool1d(identity, kernel_size=self.stride, stride=self.stride)

        # If channels mismatch, pad along channel dim
        if self.out_ch != self.in_ch:
            identity = identity.transpose(1, 2)
            ch_pad = self.out_ch - self.in_ch
            ch_left = ch_pad // 2
            ch_right = ch_pad - ch_left
            identity = F.pad(identity, (ch_left, ch_right), "constant", 0)
            identity = identity.transpose(1, 2)

        out += identity
        return out


class RSC1D(nn.Module):
    def __init__(self, in_ch, base_ch, k_size, stride,groups, blocks, num_cls, ds_gap=2,ch_up=4, use_bn=True, use_dropout=True, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.use_bn = use_bn

        self.entry_conv = conv1d_same(in_ch, base_ch, k_size, stride=1)
        self.entry_bn = nn.BatchNorm1d(base_ch)
        self.entry_relu = nn.ReLU()

        self.blocks = nn.ModuleList()
        in_ch = base_ch
        out_ch = base_ch
        for i in range(blocks):
            is_first = i == 0
            down = (i % ds_gap == 1)
            if i > 0 and i % ch_up == 0:
                out_ch = in_ch * 2
            block = ResidualBlock1D(in_ch, out_ch, k_size, stride, groups,
                                    downsample=down, use_bn=use_bn,
                                    use_dropout=use_dropout, first=is_first)
            self.blocks.append(block)
            in_ch = out_ch

        self.final_bn = nn.BatchNorm1d(out_ch)
        self.final_relu = nn.ReLU()
        self.classifier = nn.Linear(out_ch, num_cls)

    def forward(self, x):
        if self.verbose:
            print("Input:", x.shape)
        x = pad_same(x, self.entry_conv.kernel_size[0], self.entry_conv.stride[0])
        x = self.entry_conv(x)
        if self.use_bn:
            x = self.entry_bn(x)
        x = self.entry_relu(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.verbose:
                print(f"After block {i}: {x.shape}")

        if self.use_bn:
            x = self.final_bn(x)
        x = self.final_relu(x)
        x = x.mean(dim=-1)
        x = self.classifier(x)
        return x
