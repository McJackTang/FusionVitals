""" PhysNet
We repulicate the net pipeline of the orginal paper, but set the input as diffnormalized data.
orginal source:
Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks
British Machine Vision Conference (BMVC)} 2019,
By Zitong Yu, 2019/05/05
Only for research purpose, and commercial use is not allowed.
MIT License
Copyright (c) 2019
"""

import math
import pdb

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


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

        # self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

        # 新增 ConvBlock11 用于 SpO2 预测
        self.ConvBlock11 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),  # 修改此处以确保输出为 4
            nn.Flatten(),
            nn.Linear(128, 1),  # 将输出维度改为 1
            nn.Sigmoid()  # SpO2 应该在 0 到 1 之间
        )

    def forward(self, x):  # Batch_size*[3, T, 128,128]
        # print(f"x.shape: {x.shape}")                          
        x_visual = x
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)  # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x_visual6464 = self.ConvBlock3(x)  # x [32, T, 64,64]
        # x [32, T/2, 32,32]    Temporal halve
        x = self.MaxpoolSpaTem(x_visual6464)

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x_visual1616)  # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]

        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]
        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]

        # x [64, T, 1,1]    -->  groundtruth left and right - 7
        x = self.poolspa(x)
        x = self.ConvBlock10(x)  # x [1, T, 1,1]

        rPPG = x.view(batch, length)  # 保持 batch size 和长度
        # 使用新增的 ConvBlock11 来计算 SpO2
        spo2 = self.ConvBlock11(rPPG.unsqueeze(1))  # 增加一个维度以适配 Conv1d
        spo2 = spo2.view(batch, 1)  # 调整 SpO2 的形状为 [batch, 4, 1]
        # 将 SpO2 预测值从 [0, 1] 映射到 [85, 100]
        spo2 = spo2 * 15 + 85
        # print("\nspo2.shape: ", spo2.shape)
        # print("\nspo2.value: ", spo2)

        return rPPG, spo2, x_visual, x_visual3232, x_visual1616
