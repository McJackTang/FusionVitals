import math
import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple

class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),  # 压缩通道
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),  # 恢复通道
            nn.Sigmoid()  # 激活函数，生成通道权重
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 压缩空间维度
        y = self.fc(y).view(b, c, 1, 1, 1)  # 恢复形状
        return x * y  # 通道注意力加权

# class FusionNet(nn.Module):
#     def __init__(self):
#         super(FusionNet, self).__init__()
#         self.conv1 = nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm3d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm3d(64)
#         self.conv3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)  

#     def forward(self, x1, x2):
#         x = torch.cat((x1, x2), dim=1)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.conv3(x)  
#         return x

class FusionNet(nn.Module):
    def __init__(self, feature_channels=64):
        super(FusionNet, self).__init__()
        self.rgb_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(feature_channels, feature_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.ir_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(feature_channels, feature_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv3d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels, feature_channels, kernel_size=3, padding=1)
        )

    def forward(self, x_rgb, x_ir):
        # Gating: 输出每个模态的权重 [B, 1, 1, 1, 1]
        gate_rgb = self.rgb_gate(x_rgb)
        gate_ir = self.ir_gate(x_ir)

        fused = gate_rgb * x_rgb + gate_ir * x_ir

        fused = self.fuse_conv(fused)
        return fused
    
# class FusionNet(nn.Module):
#     def __init__(self, ir_weight=0.3):  
#         super(FusionNet, self).__init__()
#         self.ir_weight = ir_weight  

#         # self.rgb_se = SEBlock(64)
#         # self.ir_se = SEBlock(64)

#         # self.conv1 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.conv1 = nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm3d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm3d(64)
#         self.conv3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)

#     def forward(self, x1, x2):

#         # x1 = self.rgb_se(x1)      # [B, 64, T, H, W]
#         # x2 = self.ir_se(x2)
#         # x2 = self.ir_weight * x2  
#         # x = x1 + x2  # [B, 64, 64, 18, 18]
#         x = torch.cat((x1, x2), dim=1)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.conv3(x)  
#         return x
    
class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=128):
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            SEBlock(16)
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            SEBlock(32)
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )


        self.rppg_branch = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),            
            nn.ELU(),
            nn.AdaptiveAvgPool3d((frames, 1, 1)),
            nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0),
        )

        self.rr_branch = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool3d((frames, 1, 1)),
            nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0) 
        )
        
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
        [batch, channel, length, width, height] = x1.shape   
        if x2 is not None:
            # x1_visual = self.share_m(x1)
            # x2_visual = self.share_m(x2)
            x1_visual = self.encode_video(x1)
            x2_visual = self.encode_video(x2)
            # print(f"x1_visual.shape: {x1_visual.shape}")
            # print(f"x2_visual.shape: {x2_visual.shape}")
            
            x = self.fusion_net(x1_visual, x2_visual)
            # x = self.encode_video_x(x)

            # x = self.fusion_net(x1, x2)
            # x = self.encode_video_x(x)
            # x = self.encode_video(x)
        else:
            # x = self.share_m(x1)
            # rPPG = self.encode_video_rppg(x)
            # rr = self.encode_video_rr(x)
            x = self.encode_video(x1)
        # print(f"x.shape: {x.shape}")
        

        rPPG = self.rppg_branch(x)
        rPPG = rPPG.view(batch, length)
        rr = self.rr_branch(x)
        rr = rr.view(batch, length)

        spo2 = self.ConvBlock11(rPPG.unsqueeze(1)) 
        spo2 = spo2.view(batch, 1)
        spo2 = spo2 * 15 + 85

        return rPPG, spo2, rr

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
        return x
    
    def share_m(self, x):
        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock4(x)
        return x

    def encode_video_x(self, x):
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock8(x)
        return x