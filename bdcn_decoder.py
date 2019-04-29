import numpy as np
import torch
import torch.nn as nn

import vgg16_c


class Decoder(nn.Module):
    def __init__(self, pretrain=None, logger=None, in_dim=4224, rate=4):
        super(BDCN_ti, self).__init__()
        self.pretrain = pretrain

        self.ti_readout_1 = nn.Conv2d(in_dim, 16, (1, 1), stride=1, bias=True) # originally 105
        self.ti_activation_1 = nn.ReLU(inplace=True)
        self.ti_readout_2 = nn.Conv2d(16, 2, (1, 1), stride=1, bias=True)

        self._initialize_weights(logger)

    def forward(self, x):

        # x should be (b, ch, 1, 1)
        out = self.ti_readout_1(x)
        out = self.ti_readout_2(self.ti_activation_1(out))

        return out

    def _initialize_weights(self, logger=None):
        for name, param in self.state_dict().items():
            if self.pretrain and 'features' in name:
                continue
            if logger:
                logger.info('init params %s ' % name)
            if 'bias' in name:
                param.zero_()
            else:
                param.normal_(0, 0.01)
        # print self.conv1_1_down.weight


if __name__ == '__main__':
    model = Decoder('./caffemodel2pytorch/vgg16.pth')
    a=torch.rand((2,3,100,100))
    a=torch.autograd.Variable(a)
    for x in model(a):
        print x.data.shape
    # for name, param in model.state_dict().items():
    #     print name, param
