import torch
import torch
import torch.nn as nn
import torchvision


class DHSNet(nn.Module):

    def __init__(self):
        super(DHSNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
            )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
            )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
            )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
            )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
            )

        # Copy parameters from pretrained vgg16 network
        self.__copy_param()

        self.fc = nn.Linear(14*14*512, 784)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.rcl1 = RCL(512)
        self.rcl2 = RCL(256)
        self.rcl3 = RCL(128)
        self.rcl4 = RCL(64)
        return

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c5 = c5.view(1, -1)

        x = self.fc(c5)
        x = x.view(-1, 1, 28, 28)

        x = self.rcl1(x, c4)
        x = self.upsample(x)

        x = self.rcl2(x, c3)
        x = self.upsample(x)

        x = self.rcl3(x, c2)
        x = self.upsample(x)

        x = self.rcl4(x, c1)

        return x

    def __copy_param(self):

        # Get pretrained vgg network
        vgg16 = torchvision.models.vgg16_bn(pretrained=True)

        # Concatenate layers of generator network
        DGG_features = list(self.conv1.children())
        DGG_features.extend(list(self.conv2.children()))
        DGG_features.extend(list(self.conv3.children()))
        DGG_features.extend(list(self.conv4.children()))
        DGG_features.extend(list(self.conv5.children()))
        DGG_features = nn.Sequential(*DGG_features)

        # Copy parameters from vgg16
        for layer_1, layer_2 in zip(vgg16.features, DGG_features):
            if(isinstance(layer_1, nn.Conv2d) and
               isinstance(layer_2, nn.Conv2d)):
                assert layer_1.weight.size() == layer_2.weight.size()
                assert layer_1.bias.size() == layer_2.bias.size()
                layer_2.weight.data = layer_1.weight.data
                layer_2.bias.data = layer_1.bias.data
        return


class RCL(nn.Module):

    def __init__(self, in_channels):
        super(RCL, self).__init__()
        self.conv_pre = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1, padding=0),
            nn.Sigmoid()
            )

        self.conv_t0 = nn.Sequential(
                nn.Conv2d(65, 64, 3, padding=1),
                nn.ReLU(inplace=True)
                )
        self.conv_t1 = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True)
                )
        self.conv_t2 = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True)
                )
        self.conv_t3 = nn.Sequential(
                nn.Conv2d(64, 1, 1, padding=0),
                nn.Sigmoid()
                )
        return

    def forward(self, coarse_mask, feature_map):
        f = self.conv_pre(feature_map)
        x = torch.cat((coarse_mask, f), 1)
        x = self.conv_t0(x)
        x = self.conv_t1(x)
        x = self.conv_t2(x)
        x = self.conv_t3(x)

        return x
