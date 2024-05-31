import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedBilinearPooling(nn.Module):
    def __init__(self, channels):
        super(FactorizedBilinearPooling, self).__init__()
        self.channels = channels
        self.maxpool = nn.MaxPool3d(2)
        self.avgpool = nn.AvgPool3d(2)
        self.conv1 = nn.Conv3d(channels, channels // 2, kernel_size=(1, 1, 1))
        self.conv2 = nn.Conv3d(channels // 2, channels, kernel_size=(1, 1, 1))

    def forward(self, x, y, z):
        # x,y has dimensions (b, c, h, w, d)
        b, c, h, w, d = x.size()
        x_mp = self.maxpool(x)
        y_mp = self.maxpool(y)
        z_mp = self.maxpool(z)

        x_ap = self.avgpool(x)
        y_ap = self.avgpool(y)
        z_ap = self.avgpool(z)

        x = x_mp + x_ap
        y = y_mp + y_ap
        z = z_mp + z_ap

        x = x.view(b, c, -1)
        y = y.view(b, c, -1)
        z = z.view(b, c, -1)

        xy = x + y
        xz = x + z
        yz = y + z
        # compute outer product of x, y
        outer_xy = torch.einsum('bci,bcj->bcij', xy, xy)
        outer_xz = torch.einsum('bci,bcj->bcij', xz, xz)
        outer_yz = torch.einsum('bci,bcj->bcij', yz, yz)

        outer = outer_xy + outer_yz + outer_xz
        outer = outer.contiguous().view(b, c, -1)

        # outer = outer.view(b, c, h*h, w*w, d*d)
        # pooled = self.conv2(torch.relu(self.conv1(outer)))
        pooled = outer.sum(dim=2)
        pooled = F.normalize(pooled, dim=-1)

        return pooled