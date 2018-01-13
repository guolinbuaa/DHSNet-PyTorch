import torch
import torch.nn as nn
import torchvision
import functools
import pdb
from torch.nn import init


class RCL_Module(nn.Module):
    def __init__(self,in_channels):
        super(RCL_Module, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(65,64,3,padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,3,padding=1)
    def forward(self,x,smr):
        out1 = self.conv1(x)
        out1 = self.sigmoid(out1)
        out2 = self.sigmoid(smr)
        out = torch.cat((out1,out2),1)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn(out) #out_share
        out_share = out
        for i in range(3):
            out = self.conv3(out)
            out = torch.add(out,out_share)
            out = self.relu(out)
            out = self.bn(out)
        return out

class Feature(nn.Module):
    def __init__(self,block):
        super(Feature,self).__init__()
        self.vgg_pre = []
        self.block = block
        #vggnet
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.vgg_pre.append(nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),#conv1_2,index=[0,3]
        ))
        self.vgg_pre.append(nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),   #conv2_2,index=[1][4]
        ))
        self.vgg_pre.append(nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),  #conv3_3,index=[2][6]
        ))
        self.vgg_pre.append(nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),  #conv4_3,index=[3][6]
        ))
        self.vgg_pre.append(nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            # conv5 features
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
        ))
        self.fc= nn.Linear(14*14*512,784)
        self.layer1 = block(512)
        self.layer2 = block(256)
        self.layer3 = block(128)
        self.layer4 = block(64)
        self.features = nn.ModuleList(self.vgg_pre)
        vgg16 = torchvision.models.vgg16(pretrained=True)
        L_vgg16 = list(vgg16.features)

        self.L_self = functools.reduce(lambda x, y: list(x) + list(y), self.features)
        # L_self is a unfolded list of self.features,len()=30
        for l1, l2 in zip(L_vgg16[0:], self.L_self[0:]):
            if (isinstance(l1, nn.Conv2d) and
                    isinstance(l2, nn.Conv2d)):

                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            if (isinstance(l1, nn.BatchNorm2d) and
                    isinstance(l2, nn.BatchNorm2d)):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

    def forward(self,x):
        for f in self.features:
            x = f(x)

        x = x.view(1, -1)
        x = self.fc(x) #generate the SMRglobal
        x = x.view(1,28,-1)
        x = x.unsqueeze(0)
        x = self.layer1.forward(self.features[3][6],x)
        x = self.upsample(x)
        x = self.layer2.forward(self.features[2][6], x)
        x = self.upsample(x)
        x = self.layer3.forward(self.features[1][4], x)
        x = self.upsample(x)
        x = self.layer4.forward(self.features[0][3], x)
        return x
