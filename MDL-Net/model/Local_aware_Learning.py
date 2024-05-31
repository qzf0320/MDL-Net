import torch
import torch.nn as nn


class LocalAwareLearning(nn.Module):
    def __init__(self, in_chans1, in_chans2, in_chans3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_chans1, in_chans2, 3, 2, 1)
        self.conv2 = nn.Conv3d(in_chans2, in_chans3, 3, 2, 1)
        self.conv3 = nn.Conv3d(in_chans3, in_chans3*2, 3, 2, 1)
        self.s1 = nn.Sequential(
            nn.Conv3d(in_chans1, in_chans1, 1),
            nn.Sigmoid(),
        )
        self.s2 = nn.Sequential(
            nn.Conv3d(in_chans2 * 2, in_chans2, 1),
            nn.Sigmoid(),
        )
        self.s3 = nn.Sequential(
            nn.Conv3d(in_chans3 * 2, in_chans3, 1),
            nn.Sigmoid(),
        )

    def forward(self, s1, s2, s3):
        s1_0 = self.s1(s1)
        s1 = s1 * s1_0
        s1 = self.conv1(s1)

        s2_0 = torch.cat((s2, s1), 1)
        s2_1 = self.s2(s2_0)
        s2 = s2 * s2_1
        s2 = self.conv2(s2)

        s3_0 = torch.cat((s3, s2), 1)
        s3_1 = self.s3(s3_0)
        s3 = s3 * s3_1
        s3 = self.conv3(s3)
        out = s3

        return out