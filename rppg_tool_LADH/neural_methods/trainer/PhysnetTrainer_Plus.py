import math
import pdb

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv1 = nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=128):
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

        self.ConvBlock11 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.fusion_net = FusionNet()

    def forward(self, x1, x2=None):  # Batch_size*[3, T, 128,128]
        if x2 is not None:
            x1_visual = self.encode_video(x1)
            x2_visual = self.encode_video(x2)
            x = self.fusion_net(x1_visual, x2_visual)
        else:
            x = self.encode_video(x1)

        rPPG = x.view(x.size(0), -1)  # 保持 batch size 和长度
        spo2 = self.ConvBlock11(rPPG.unsqueeze(1))
        spo2 = spo2.view(x.size(0), 1)
        spo2 = spo2 * 20 + 80

        return rPPG, spo2

    def encode_video(self, x):
        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock8(x)
        x = self.ConvBlock9(x)
        x = self.upsample(x)
        x = self.upsample2(x)
        x = self.poolspa(x)
        x = self.ConvBlock10(x)
        return x
