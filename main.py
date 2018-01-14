import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import dut_omron_dataset
import models.DHSNet as DHSNet


def main(use_gpu, root, root_gt, batch_size, epoch_num):
    if use_gpu and not torch.cuda.is_available():
        print('Cannot use gpu: CUDA not available')
        use_gpu = False

    dut_omron = dut_omron_dataset.get_dut_omron(
        root=root,
        root_gt=root_gt,
        batch_size=batch_size
        )

    network = DHSNet.DHSNet()
    if use_gpu:
        network = network.cuda()

    optimizer_feature = torch.optim.Adam(network.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch_i in range(epoch_num):
        data_iter = iter(dut_omron)
        for batch_i in range(len(dut_omron)):
            inputs, gts = data_iter.next()
            inputs = Variable(inputs)
            gts = Variable(gts)
            if use_gpu:
                inputs = inputs.cuda()
                gts = gts.cuda()

            outputs = network(inputs)
            loss = criterion(outputs, gts)
            loss.backward()
            optimizer_feature.step()

            print('e{0:d}_b{1:d}: loss: {2:f}'.format(
                                                epoch_i, batch_i, loss.data[0]
                                                ))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root', required=False, default='data/DUT-OMRON/OMRON-Image/'
        )
    parser.add_argument(
        '--root_gt', required=False, default='data/DUT-OMRON/OMRON-GT'
        )
    parser.add_argument(
        '--batch_size', required=False, type=int, default=1
        )
    parser.add_argument(
        '--iter_num', required=False, type=int, default=10
        )
    opt = parser.parse_args()

    main(
        use_gpu=False,
        root=opt.root,
        root_gt=opt.root_gt,
        batch_size=opt.batch_size,
        epoch_num=opt.iter_num
        )
