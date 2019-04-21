import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import argparse
import time
import re
import os
import sys
import bdcn
from datasets.dataset import BSDS_crops, Multicue_crops, Tilt_illusion
import cfg
import log
import cv2

def l2_loss_center(out, labels):
    _, _, h, w = out.size()
    out_center = out[:, :, h//2, w//2]
    return out_center, torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')(out_center, labels)

def l2_loss(out, labels):
    # ipdb > out.shape
    # torch.Size([B, 2, 1, 1])
    # ipdb > labels.shape
    # torch.Size([B, 1, 2])
    labels = labels.permute(0,2,1).unsqueeze(3)
    return torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')(out, labels)


def train(model, args):
    # Configure datasets
    # import ipdb;
    # ipdb.set_trace()
    print(args.dataset)
    crop_size = args.crop_size
    if 'tiltillusion' in args.dataset:
        data_root = '/media/data_cifs/tilt_illusion'
        # Construct data loader
        test_img = Tilt_illusion(data_root, type='test', test_mode=True,
                                  max_examples=args.max_test_examples, scale=[0.4], crop_size=crop_size)
        testloader = torch.utils.data.DataLoader(test_img,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=5)
    else:
        raise ValueError('dataset should be tiltillusion.')

    # Configure train
    logger = args.logger
    start_step = 1
    mean_loss = []
    cur = 0
    pos = 0
    data_iter = iter(testloader)
    iter_per_epoch = len(testloader)
    start_time = time.time()
    if args.cuda:
        model.cuda()
    model.eval() # same as model.train(mode=False)

    import matplotlib.pyplot as plt

    for step in xrange(start_step, args.max_test_examples/(args.iter_size*args.batch_size) + 1):
        batch_loss = 0
        for i in xrange(args.iter_size):
            if cur == iter_per_epoch:
                cur = 0
                data_iter = iter(testloader)
            images, labels, meta = next(data_iter)
            # import ipdb;ipdb.set_trace()
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

            out = model(images)
            import ipdb;ipdb.set_trace()
            # out.squeeze().shape = labels.squeeze().shape = [10,2]
            # import ipdb;
            # ipdb.set_trace()

            loss = l2_loss(out, labels)

            # import ipdb;ipdb.set_trace()
            batch_loss += loss.item()
            cur += 1
        # update parameter
        if len(mean_loss) < args.average_loss:
            mean_loss.append(batch_loss)
        else:
            mean_loss[pos] = batch_loss
            pos = (pos + 1) % args.average_loss
        if step % args.display == 0:
            tm = time.time() - start_time
            logger.info('iter: %d, loss: %f, time using: %f(%fs/batch)' % (step,
               np.mean(mean_loss), tm, tm/(args.iter_size*args.display)))
            start_time = time.time()

def main():
    args = parse_args()
    logger = log.get_logger(args.log)
    args.logger = logger
    logger.info('*'*80)
    logger.info('the args are the below')
    logger.info('*'*80)
    for x in args.__dict__:
        logger.info(x+','+str(args.__dict__[x]))
    logger.info(cfg.config[args.dataset])
    logger.info('*'*80)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(long(time.time()))
    model = bdcn.BDCN_ti(pretrain=None, logger=logger)
    model.initialize_ti_weights()
    if args.complete_pretrain:
        model.load_state_dict(torch.load(args.complete_pretrain))
    logger.info(model)
    train(model, args)

def parse_args():
    parser = argparse.ArgumentParser(description='Train BDCN for different args')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config.keys(),
        default='bsds500', help='The dataset to train')
    parser.add_argument('--max-test-examples', type=int, default=None,
        help='(jk) max iters to test network, default is None (200 for BSDS)')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('--iter-size', type=int, default=10,
        help='iter size equal to the batch size, default 10')
    parser.add_argument('--average-loss', type=int, default=50,
        help='smoothed loss, default is 50')
    parser.add_argument('--step-size', type=int, default=10000,
        help='the number of iters to decrease the learning rate, default is 10000')
    parser.add_argument('--display', type=int, default=20,
        help='how many iters display one time, default is 20')
    parser.add_argument('-l', '--log', type=str, default='log.txt',
        help='the file to store log, default is log.txt')
    parser.add_argument('--batch-size', type=int, default=1,
        help='batch size of one iteration, default 1')
    parser.add_argument('--crop-size', type=int, default=None,
        help='the size of image to crop, default not crop')
    parser.add_argument('--complete-pretrain', type=str, default=None,
        help='finetune on the complete_pretrain, default None')
    return parser.parse_args()

if __name__ == '__main__':
    main()


