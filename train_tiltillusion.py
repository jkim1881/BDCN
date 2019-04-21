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

def adjust_learning_rate(optimizer, steps, step_size, gamma=0.1, logger=None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma
        if logger:
            logger.info('%s: %s' % (param_group['name'], param_group['lr']))


def cross_entropy_loss2d(inputs, targets, cuda=False, balance=1.1):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()
    weights = np.zeros((n, c, h, w))
    for i in xrange(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * balance / valid
    weights = torch.Tensor(weights)
    if cuda:
        weights = weights.cuda()
    inputs = F.sigmoid(inputs)
    loss = nn.BCELoss(weights, size_average=False)(inputs, targets)
    return loss

def l2_loss(out, labels):
    # ipdb > out.shape
    # torch.Size([B, 2, 1, 1])
    # ipdb > labels.shape
    # torch.Size([B, 1, 2])
    labels = labels.permute(0,2,1).unsqueeze(3)
    pos = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')(out, labels)
    neg = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')(-out, labels)
    return torch.min(pos,neg) # mirror-symmetric loss

def train(model, args):
    # Configure datasets
    # import ipdb;
    # ipdb.set_trace()
    print(args.dataset)
    crop_size = args.crop_size
    if 'tiltillusion' in args.dataset:
        data_root = '/media/data_cifs/tilt_illusion'
        # Construct data loader
        train_img = Tilt_illusion(data_root, type='train',
                                  max_examples=args.max_training_examples, scale=[0.3], crop_size=crop_size)
        trainloader = torch.utils.data.DataLoader(train_img,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=5)
        val_img = Tilt_illusion(data_root, type='test',
                            crop_size=crop_size, scale=[0.3])
        valloader = torch.utils.data.DataLoader(val_img,
                                                batch_size=args.batch_size, shuffle=False, num_workers=5)
    else:
        raise ValueError('dataset should be tiltillusion.')

    # Configure train
    params_dict = dict(model.named_parameters())
    base_lr = args.base_lr
    weight_decay = args.weight_decay
    logger = args.logger
    params = []
    for key, v in params_dict.items():
        if 'ti' in key:
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr, 'weight_decay': weight_decay*1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr, 'weight_decay': weight_decay*0, 'name': key}]
    optimizer = torch.optim.SGD(params, momentum=args.momentum,
        lr=args.base_lr, weight_decay=args.weight_decay)
    start_step = 1
    mean_loss = []
    cur = 0
    val_cur = 0
    pos = 0
    data_iter = iter(trainloader)
    iter_per_epoch = len(trainloader)
    val_data_iter = iter(valloader)
    val_iter_per_epoch = len(valloader)
    logger.info('*'*40)
    logger.info('train images in all are %d ' % iter_per_epoch)
    logger.info('*'*40)
    for param_group in optimizer.param_groups:
        if logger:
            logger.info('%s: %s' % (param_group['name'], param_group['lr']))
    start_time = time.time()
    if args.cuda:
        model.cuda()
    if args.resume:
        logger.info('resume from %s' % args.resume)
        state = torch.load(args.resume)
        start_step = state['step']
        optimizer.load_state_dict(state['solver'])
        model.load_state_dict(state['param'])
    model.train()
    batch_size = args.iter_size * args.batch_size

    import matplotlib.pyplot as plt

    for step in xrange(start_step, args.max_iter + 1):
        optimizer.zero_grad()
        batch_loss = 0
        for i in xrange(args.iter_size):
            if cur == iter_per_epoch:
                cur = 0
                data_iter = iter(trainloader)
            images, labels = next(data_iter)
            # import ipdb;ipdb.set_trace()
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

            out = model(images)
            # import ipdb;
            # ipdb.set_trace()
            loss = l2_loss(out, labels)

            if ((step ==1) or (step == 100)) and args.display_imgs==1:
                batchid = 0
                img_min = np.min(np.array(images.cpu()[batchid, :, :, :].flatten()))
                img_max = np.max(np.array(images.cpu()[batchid, :, :, :].flatten()))
                img_transposed = (np.transpose(np.array(images.cpu()[batchid, :, :, :]), (1, 2, 0)) - img_min) / (
                    img_max - img_min)
                plt.subplot(121);
                plt.imshow(img_transposed);
                plt.subplot(122);
                plt.plot(out.squeeze().detach().cpu()[batchid,1], out.squeeze().detach().cpu()[batchid,0], marker='x')
                plt.plot(labels.squeeze().detach().cpu()[batchid, 1], labels.squeeze().detach().cpu()[batchid, 0], marker='o')
                plt.xlim(-2,2)
                plt.ylim(-2,2)
                plt.show()

            # import ipdb;ipdb.set_trace()
            loss.backward()
            batch_loss += loss.item()
            cur += 1
        # update parameter
        optimizer.step()
        if len(mean_loss) < args.average_loss:
            mean_loss.append(batch_loss)
        else:
            mean_loss[pos] = batch_loss
            pos = (pos + 1) % args.average_loss
        if step % args.step_size == 0:
            adjust_learning_rate(optimizer, step, args.step_size, args.gamma)
        if step % args.snapshots == 0:
            logger.info('iter: %d, Saving snapshot.....'%(step))
            torch.save(model.state_dict(), '%s/bdcn_%d.pth' % (args.param_dir, step))
            state = {'step': step+1,'param':model.state_dict(),'solver':optimizer.state_dict()}
            torch.save(state, '%s/bdcn_%d.pth.tar' % (args.param_dir, step))
        if step % args.display == 0:
            tm = time.time() - start_time
            logger.info('iter: %d, lr: %e, loss: %f, time using: %f(%fs/batch)' % (step,
                optimizer.param_groups[0]['lr'], np.mean(mean_loss), tm, tm/(args.iter_size*args.display)))
            start_time = time.time()

        # (jk) RUN VALIDATION ON SPECIFIED ITERATIONS
        if ((step % args.validation_period) == 0) and (step > 0):
            val_mean_loss = []
            for val_step in xrange(args.validation_iters):
                batch_loss = 0
                for i in xrange(args.iter_size):
                    if val_cur == val_iter_per_epoch:
                        val_cur = 0
                        val_data_iter = iter(valloader)
                    images, labels = next(val_data_iter)
                    if args.cuda:
                        images, labels = images.cuda(), labels.cuda()
                    images, labels = Variable(images), Variable(labels)

                    out = model(images)
                    loss = l2_loss(out, labels)

                    if (step >= 100) and (val_step==0) and args.display_imgs==1:
                        batchid = 0
                        img_min = np.min(np.array(images.cpu()[batchid, :, :, :].flatten()))
                        img_max = np.max(np.array(images.cpu()[batchid, :, :, :].flatten()))
                        img_transposed = (np.transpose(np.array(images.cpu()[batchid, :, :, :]),
                                                       (1, 2, 0)) - img_min) / (
                                             img_max - img_min)
                        plt.subplot(121);
                        plt.imshow(img_transposed);
                        plt.subplot(122);
                        plt.plot(out.squeeze().detach().cpu()[batchid, 1], out.squeeze().detach().cpu()[batchid, 0],
                                 marker='x')
                        plt.plot(labels.squeeze().detach().cpu()[batchid, 1],
                                 labels.squeeze().detach().cpu()[batchid, 0], marker='o')
                        plt.xlim(-2, 2)
                        plt.ylim(-2, 2)
                        plt.show()

                    batch_loss += loss.item()
                    val_cur += 1
                val_mean_loss.append(batch_loss)
            # Report
            logger.info('>>> Val over %d images, loss: %f' % (args.validation_iters*batch_size,
                                                              np.mean(val_mean_loss)))


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
    if not os.path.exists(args.param_dir):
        os.mkdir(args.param_dir)
    torch.manual_seed(long(time.time()))
    model = bdcn.BDCN_ti(pretrain=args.pretrain, logger=logger)
    if args.complete_pretrain:
        model.load_state_dict(torch.load(args.complete_pretrain))
    model.initialize_ti_weights()
    logger.info(model)
    train(model, args)

def parse_args():
    parser = argparse.ArgumentParser(description='Train BDCN for different args')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config.keys(),
        default='bsds500', help='The dataset to train')
    parser.add_argument('--max-training-examples', type=int, default=None,
        help='(jk) max iters to train network, default is None (200 for BSDS)')
    parser.add_argument('--param-dir', type=str, default='params',
        help='the directory to store the params')
    parser.add_argument('--lr', dest='base_lr', type=float, default=1e-6,
        help='the base learning rate of model')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
        help='the momentum')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('--weight-decay', type=float, default=0.0002,
        help='the weight_decay of net')
    parser.add_argument('-r', '--resume', type=str, default=None,
        help='whether resume from some, default is None')
    parser.add_argument('-p', '--pretrain', type=str, default=None,
        help='init net from pretrained model default is None')
    parser.add_argument('--max-iter', type=int, default=40000,
        help='max iters to train network, default is 40000')
    parser.add_argument('--validation-period', type=int, default=5000,
        help='(jk) validation period, default is 5000')
    parser.add_argument('--validation-iters', type=int, default=50,
        help='(jk) iterations per val, default is 10 (50*10 = 500 imgs)')
    parser.add_argument('--iter-size', type=int, default=10,
        help='iter size equal to the batch size, default 10')
    parser.add_argument('--average-loss', type=int, default=50,
        help='smoothed loss, default is 50')
    parser.add_argument('-s', '--snapshots', type=int, default=1000,
        help='how many iters to store the params, default is 1000')
    parser.add_argument('--step-size', type=int, default=10000,
        help='the number of iters to decrease the learning rate, default is 10000')
    parser.add_argument('--display', type=int, default=20,
        help='how many iters display one time, default is 20')
    parser.add_argument('-b', '--balance', type=float, default=1.1,
        help='the parameter to balance the neg and pos, default is 1.1')
    parser.add_argument('-l', '--log', type=str, default='log.txt',
        help='the file to store log, default is log.txt')
    parser.add_argument('-k', type=int, default=1,
        help='the k-th split set of multicue')
    parser.add_argument('--batch-size', type=int, default=1,
        help='batch size of one iteration, default 1')
    parser.add_argument('--crop-size', type=int, default=None,
        help='the size of image to crop, default not crop')
    parser.add_argument('--yita', type=float, default=None,
        help='the param to operate gt, default is data in the config file')
    parser.add_argument('--complete-pretrain', type=str, default=None,
        help='finetune on the complete_pretrain, default None')
    parser.add_argument('--side-weight', type=float, default=0.5,
        help='the loss weight of sideout, default 0.5')
    parser.add_argument('--fuse-weight', type=float, default=1.1,
        help='the loss weight of fuse, default 1.1')
    parser.add_argument('--gamma', type=float, default=0.1,
        help='the decay of learning rate, default 0.1')
    parser.add_argument('--display-imgs', type=int, default=0,
        help='(jk) display imgs at 1st and 100th iteration (val on the 100th)')
    return parser.parse_args()

if __name__ == '__main__':
    main()


