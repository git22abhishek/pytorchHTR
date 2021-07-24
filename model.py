import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock

import numpy as np


def downsample(chan_in, chan_out, stride, pad=0):

    return nn.Sequential(
        nn.Conv2d(chan_in, chan_out, kernel_size=1, stride=stride, bias=False,
                  padding=pad),
        nn.BatchNorm2d(chan_out)
    )


class CRNNModel(nn.Module):

    def __init__(self, vocab_size, time_steps, zero_init_residual=False):
        super(CRNNModel, self).__init__()
        self.vocab_size = vocab_size
        self.time_steps = time_steps

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2,
                               bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(*[BasicBlock(64, 64) for i in range(0, 2)])
        self.layer2 = nn.Sequential(*[BasicBlock(64, 128, stride=2,
                                      downsample=downsample(64, 128, 2))
                                      if i == 0 else BasicBlock(128, 128)
                                      for i in range(0, 2)])
        self.layer3 = nn.Sequential(*[BasicBlock(128, 256, stride=(1, 2),
                                      downsample=downsample(128, 256, (1, 2)))
                                      if i == 0 else BasicBlock(256, 256)
                                      for i in range(0, 2)])
        self.layer4 = nn.Sequential(*[BasicBlock(256, 512, stride=(1, 2),
                                      downsample=downsample(256, 512, (1, 2)))
                                      if i == 0 else BasicBlock(512, 512)
                                      for i in range(0, 2)])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(time_steps, 2))

        self.lstm = nn.LSTM(input_size=1024, hidden_size=512, num_layers=2,
                            bidirectional=True, batch_first=True)

        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, vocab_size + 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init_constant_(m.bn2.weight, 0)

    def forward(self, xb):

        out = self.maxpool(self.bn1(self.relu(self.conv1(xb))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)

        # print("CNN output before transpose:", out.shape)
        # out = out.squeeze(dim=3).transpose(1, 2)
        out = out.permute(0, 2, 3, 1)
        out = out.reshape(out.size(0,), out.size(1), -1)
        # print("CNN output after transpose:", out.shape)

        out, _ = self.lstm(out)
        # print("LSTM output:", out.shape)
        out = self.fc1(out)
        # print("FC1 output:", out.shape)
        out = self.fc2(out)
        # print("FC2 output:", out.shape)

        return out
