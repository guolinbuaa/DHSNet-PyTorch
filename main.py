import gc
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import torchvision
from dataset import MyData, MyTestData
from model import Feature
from model import RCL_Module
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import time
import matplotlib.pyplot as plt

train_root = '/home/gwl/datasets/DUT-OMRON/DATASET'  # training dataset
val_root = '/home/zeng/data/datasets/saliency_Dataset/ECSSD'  # validation dataset
check_root = './parameters'  # save checkpoint parameters
val_output_root = './validation'  # save validation results
bsize = 1  # batch size
iter_num = 20  # training iterations


std = [.229, .224, .225]
mean = [.485, .456, .406]

os.system('rm -rf ./runs/*')
writer = SummaryWriter('./runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))

if not os.path.exists('./runs'):
    os.mkdir('./runs')

if not os.path.exists(check_root):
    os.mkdir(check_root)

if not os.path.exists(val_output_root):
    os.mkdir(val_output_root)

# models
feature = Feature(RCL_Module)
feature.cuda()

train_loader = torch.utils.data.DataLoader(
    MyData(train_root, transform=True),
    batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)
#return image,gt

val_loader = torch.utils.data.DataLoader(
    MyTestData(train_root, transform=True),
    batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

criterion = nn.BCEWithLogitsLoss()
optimizer_feature = torch.optim.Adam(feature.parameters(), lr=1e-4)
istep = 0


# def validation(val_loader, output_root, feature):
#     if not os.path.exists(output_root):
#         os.mkdir(output_root)
#     for ib, (data, img_name, img_size) in enumerate(val_loader):
#         print(ib)
#         start = time.time()
#         prior = prior.unsqueeze(1)
#         data = torch.cat((data, prior), 1)
#
#         inputs = Variable(data).cuda()
#
#         feats = feature(inputs)
#         feats = feats[-3:]
#         feats = feats[::-1]
#         msk = deconv(feats)
#
#         msk = functional.upsample(msk, scale_factor=4)
#
#         msk = functional.sigmoid(msk)
#
#         mask = msk.data[0, 0].cpu().numpy()
#         plt.imsave(os.path.join(output_root, img_name[0]+'.png'), mask, cmap='gray')


for it in range(iter_num):
    for ib, (data, lbl) in enumerate(train_loader):
        inputs = Variable(data).cuda()
        lbl = Variable(lbl.unsqueeze(1)).cuda()
        msk = feature.forward(inputs)
        loss = criterion(msk, lbl)
        feature.zero_grad()
        loss.backward()
        optimizer_feature.step()

        print('loss: %.4f (epoch: %d, step: %d)' % (loss.data[0], it, ib))

        # visulize
        # image = make_image_grid(inputs.data[:, :3], mean, std)
        # writer.add_image('Image', torchvision.utils.make_grid(image), ib)
        # msk = functional.sigmoid(msk)
        # mask1 = msk.data
        # mask1 = mask1.repeat(1, 3, 1, 1)
        # writer.add_image('Image2', torchvision.utils.make_grid(mask1), ib)
        # print('loss: %.4f (epoch: %d, step: %d)' % (loss.data[0], it, ib))
        # writer.add_scalar('M_global', loss.data[0], istep)
        # istep += 1

        # del inputs, msk, lbl, loss, feats, mask1, image
        # gc.collect()
        if ib % 1000 == 0:
            filename = ('%s/feature-epoch-%d-step-%d.pth' % (check_root, it, ib))
            torch.save(feature.state_dict(), filename)
            print('save: (epoch: %d, step: %d)' % (it, ib))
    # validation(val_loader, '%s/%d'%(val_output_root, it), feature, deconv)


