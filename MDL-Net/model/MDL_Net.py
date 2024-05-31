import time
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from einops import rearrange
# from module.scale_fusion import LocalAwareLearning
from model.Disease_incuded_ROI import Disease_Guide_ROI
from model.Latent_Space_aware_Learning import FactorizedBilinearPooling
from model.Local_aware_Learning import LocalAwareLearning
from model.Self_adaptive_Transformer import SelfAdaptiveTransformer
from thop import profile


def get_inplanes():
    return [16, 32, 64, 128]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MDL_Net(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 iter=3,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=2,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.global_fusion = SelfAdaptiveTransformer(hidden_size=block_inplanes[3], img_size=(128, 128, 128),
                                                       patch_size=(32, 32, 32), num_heads=4, attention_dropout_rate=0.2,
                                                       window_size=(2, 2, 2))

        self.local_fusion = LocalAwareLearning(in_chans1=block_inplanes[0] * block.expansion,
                           in_chans2=block_inplanes[1] * block.expansion,
                           in_chans3=block_inplanes[2] * block.expansion)
        
        self.fbc = FactorizedBilinearPooling(block_inplanes[3] * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.maxpool = nn.MaxPool3d(2)
        
        self.fc_roi = nn.Linear(block_inplanes[3] * block.expansion * 3, 90)
        self.fc_cls2roi = nn.Linear(n_classes, 90)
        self.roi = Disease_Guide_ROI(90, 6, i=iter)

        self.dropout = nn.Dropout(0.5)

        self.roi_fc = nn.Linear(90, n_classes)

        self.pred_fc = nn.Linear(block_inplanes[3] * block.expansion * 3, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def feature_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)

        # return out
        return s1, s2, s3, s4

    def forward(self, x):
        # Upsample and Split Multimodal Feature
        x = F.interpolate(x, [128, 128, 128], mode='trilinear')
        x1 = x[:, 0, :, :, :]
        x1 = torch.unsqueeze(x1, dim=1)    # The unsqueeze operation is to add channel of single-modal dataset
        x2 = x[:, 1, :, :, :]
        x2 = torch.unsqueeze(x2, dim=1)
        x3 = x[:, 2, :, :, :]
        x3 = torch.unsqueeze(x3, dim=1)

        # Extract single modal feature
        x1_s1, x1_s2, x1_s3, x1_s4 = self.feature_forward(x1)
        x2_s1, x2_s2, x2_s3, x2_s4 = self.feature_forward(x2)
        x3_s1, x3_s2, x3_s3, x3_s4 = self.feature_forward(x3)

        # Global-aware Learning
        x1_s4 = self.dropout(x1_s4)
        x2_s4 = self.dropout(x2_s4)
        x3_s4 = self.dropout(x3_s4)
        b, c, h, w, d = x1_s4.size()
        fusion = self.global_fusion(x1_s4, x2_s4, x3_s4)
        fusion = rearrange(fusion, 'b (h w d) c->b c h w d', h=h, w=w, d=d)
        fusion = self.avgpool(fusion)
        out_global = fusion.view(fusion.shape[0], -1)

        # Latent-space aware Learning
        out_latent = self.fbc(x1_s4, x2_s4, x3_s4)
        out = torch.cat((out_global, out_latent), dim=1)

        # Local-aware Learning
        out_s1 = x1_s1 + x2_s1 + x3_s1
        out_s2 = x1_s2 + x2_s2 + x3_s2
        out_s3 = x1_s3 + x2_s3 + x3_s3
        # out_s4 = x1_s4 + x2_s4 + x3_s4
        fusion_l = self.local_fusion(out_s1, out_s2, out_s3)
        fusion_l = self.avgpool(fusion_l)
        fusion_l = fusion_l.view(fusion_l.shape[0], -1)
        out = torch.cat((out, fusion_l), dim=1)
        class_out = self.pred_fc(out)

        # Diseased-incuded ROI Learning
        f_roi = self.fc_roi(out).unsqueeze(dim=1)
        cls_roi = self.fc_cls2roi(class_out).unsqueeze(dim=1)
        roi = self.roi(f_roi, cls_roi)
        roi_cls = self.roi_fc(roi)
        
        # Classification 
        class_out = class_out + roi_cls

        return class_out, roi


def generate_model(model_depth, in_planes, num_classes, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = MDL_Net(BasicBlock, [1, 1, 1, 1], get_inplanes(), n_input_channels=in_planes,
                      n_classes=num_classes,
                      **kwargs)
    elif model_depth == 18:
        model = MDL_Net(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_input_channels=in_planes,
                      n_classes=num_classes,
                      **kwargs)
    elif model_depth == 34:
        model = MDL_Net(BasicBlock, [3, 4, 6, 3], get_inplanes(), n_input_channels=in_planes,
                      n_classes=num_classes,
                      **kwargs)
    elif model_depth == 50:
        model = MDL_Net(Bottleneck, [3, 4, 6, 3], get_inplanes(), n_input_channels=in_planes,
                      n_classes=num_classes,
                      **kwargs)
    elif model_depth == 101:
        model = MDL_Net(Bottleneck, [3, 4, 23, 3], get_inplanes(), n_input_channels=in_planes,
                      n_classes=num_classes,
                      **kwargs)
    elif model_depth == 152:
        model = MDL_Net(Bottleneck, [3, 8, 36, 3], get_inplanes(), n_input_channels=in_planes,
                      n_classes=num_classes,
                      **kwargs)
    elif model_depth == 200:
        model = MDL_Net(Bottleneck, [3, 24, 36, 3], get_inplanes(), n_input_channels=in_planes, n_classes=num_classes,
                      **kwargs)

    return model


if __name__ == '__main__':
    '''
    The shpae of x: [b, m, h, w, d]
    m is the number of modality, i.e, if have three multimodal dataset, m is 3
    '''
    x = torch.ones(3, 3, 100, 120, 100)
    y = torch.randn(1, 90)
    time_start = time.time()
    model = generate_model(model_depth=18, in_planes=1, num_classes=3)

    output, roi = model(x)
    time_over = time.time()
    print(output.shape)
    print(roi.shape)
    flops, params = profile(model, (x,))
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    print('time:{}s'.format(time_over - time_start))
