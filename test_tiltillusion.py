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
import bdcn, bdcn_decoder
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


def orientation_diff(array1, array2):
    concat = np.concatenate((np.expand_dims(array1, axis=1),
                             np.expand_dims(array2, axis=1)), axis=1)
    diffs = np.concatenate((np.expand_dims(concat[:,0] - concat[:,1], axis=1),
                            np.expand_dims(concat[:,0] - concat[:,1] - 180, axis=1),
                            np.expand_dims(concat[:,0] - concat[:,1] + 180, axis=1)), axis=1)
    diffs_argmin = np.argmin(np.abs(diffs),axis=1)
    return [idiff[argmin] for idiff, argmin in zip(diffs, diffs_argmin)]


def cluster_points(xs, ys, stepsize):
    xss = list(xs)
    sort_args = np.array(xss).argsort()
    xss.sort()
    ys_sorted = np.array(ys)[sort_args]

    x_accumulator = []
    y_mu = []
    y_25 = []
    y_75 = []
    x_perbin = []
    y_perbin = []
    icut = -90 + stepsize

    for ix, iy in zip(xss, ys_sorted):
        if ix < icut:
            x_perbin.append(ix)
            y_perbin.append(iy)
        else:
            if len(y_perbin) > 0:
                x_accumulator.append(icut - stepsize / 2)
                y_mu.append(np.median(y_perbin))
                y_25.append(np.percentile(y_perbin, 25))
                y_75.append(np.percentile(y_perbin, 75))
            icut += stepsize
            x_perbin = []
            y_perbin = []
    return x_accumulator, y_mu, y_25, y_75


def collapse_points(cs_diff, out_gt_diff):
    cs_diff_collapsed =[]
    out_gt_diff_collapsed = []
    for ix, iy in zip(cs_diff, out_gt_diff):
        if ix < -10:
            cs_diff_collapsed.append(-ix)
            out_gt_diff_collapsed.append(-iy)
        else:
            cs_diff_collapsed.append(ix)
            out_gt_diff_collapsed.append(iy)
    return cs_diff_collapsed, out_gt_diff_collapsed


def screen(r1, lambda1, theta, r1min=None, r1max=None, lambda1min=None, lambda1max=None, thetamin=None, thetamax=None):
    if np.array(r1).size > 1:
        cond = np.ones_like(r1).astype(np.bool)
    else:
        cond = True
    if r1min is not None:
        cond = cond * (r1 > r1min)
    if r1max is not None:
        cond = cond * (r1 < r1max)
    if lambda1min is not None:
        cond = cond * (lambda1 > lambda1min)
    if lambda1max is not None:
        cond = cond * (lambda1 < lambda1max)
    if thetamin is not None:
        cond = cond * ((theta > thetamin) | (theta > thetamin+180))
    if thetamax is not None:
        cond = cond * (theta < thetamax)
    return cond

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
                                  max_examples=args.max_test_examples, scale=[0.3], crop_size=crop_size)
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

    accumulator = np.zeros((0,7))
    for step in xrange(start_step, args.max_test_examples/(args.iter_size*args.batch_size) + 1):
        batch_loss = 0
        for i in xrange(args.iter_size):
            if cur == iter_per_epoch:
                cur = 0
                data_iter = iter(testloader)
            images, labels, meta = next(data_iter) # [r1, theta1, lambda1, shift1, r2 ....]
            # import ipdb;ipdb.set_trace()
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

            out = model(images)
            out_arr = out.squeeze().cpu().detach().numpy()
            out_deg = ((np.arctan2(out_arr[:,0], out_arr[:,1]))*180/np.pi)%180
            labels_arr = labels.squeeze().cpu().detach().numpy()
            labels_deg = ((np.arctan2(labels_arr[:,0], labels_arr[:,1]))*180/np.pi)%180
            meta_arr = meta.cpu().detach().numpy()

            results = np.concatenate((np.expand_dims(meta_arr[:, 1], axis=1),
                                      np.expand_dims(meta_arr[:, 5], axis=1),
                                      np.expand_dims(out_deg,    axis=1),
                                      np.expand_dims(meta_arr[:, 0], axis=1),
                                      np.expand_dims(meta_arr[:, 2], axis=1),
                                      np.expand_dims(meta_arr[:, 3], axis=1),
                                      np.expand_dims(meta_arr[:, 7], axis=1)),
                                      axis=1)

            for i in range(results.shape[0]):
                cond = screen(meta_arr[i, 3].astype(np.float), meta_arr[i, 5].astype(np.float), meta_arr[i, 4].astype(np.float),
                              r1min=100, r1max=100 + 20, lambda1min=None, lambda1max=None,
                              thetamin=22.5, thetamax=22.5 + 45)
                print(cond)
                if cond:
                    accumulator = np.concatenate((accumulator, results[i,:]), axis=0)

            # cdegree (theta1), sdegree (theta2), ydegree, r1, lambda1, shift1, shift2
            # theta1_estimated =

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
            logger.info('iter: %d, loss: %f, time using: %f(%fs/batch)' %
                        (step, np.mean(mean_loss), tm, tm/(args.iter_size*args.display)))
            start_time = time.time()

    # plot
    import matplotlib.pyplot as plt
    import numpy.polynomial.polynomial as poly


    cs_diff = orientation_diff(accumulator[:, 0], accumulator[:, 1]) # center - surround in x axis
    out_gt_diff = orientation_diff(accumulator[:, 2], accumulator[:, 0]) # pred - gt in y axis
    cs_diff_collapsed, out_gt_diff_collapsed = collapse_points(cs_diff, out_gt_diff)
    x_list, y_mu, y_25, y_75 = cluster_points(cs_diff_collapsed, out_gt_diff_collapsed, 10)
    plt.scatter(cs_diff_collapsed, out_gt_diff_collapsed, s=25, alpha=0.1, vmin=0, vmax=180, color='black')
    plt.plot(np.arange(-90, 90, 1), [0]*np.arange(-90, 90, 1).size, color='black')
    plt.fill_between(x_list, y_25, y_75, alpha=0.5)
    plt.plot(x_list, y_mu, linewidth=3, alpha=0.5, color='red')
    plt.xlim(0, 90)
    plt.ylim(-20, 50)
    plt.show()

    f = plt.figure(figsize=(4,4))
    axarr = f.subplots(4,4) #(4, 4)
    coefs = poly.polyfit(cs_diff_collapsed, out_gt_diff_collapsed, 5)
    ffit = poly.polyval(np.arange(-90, 90, 1), coefs)
    axarr.scatter(cs_diff_collapsed, out_gt_diff_collapsed, s=40, alpha=0.25, vmin=0, vmax=180)
    # coefs = poly.polyfit(cs_diff, out_gt_diff, 5)
    # ffit = poly.polyval(np.arange(-90, 90, 1), coefs)
    # axarr[ir, ith].scatter(cs_diff, out_gt_diff, s=15, alpha=0.3, vmin=0, vmax=180)
    axarr.plot(np.arange(-90, 90, 1), ffit, linewidth=3, alpha=0.5, color='black')
    axarr.plot(np.arange(-90, 90, 1), [0] * np.arange(-90, 90, 1).size, color='black')
    axarr.set_xlim(0, 87)
    axarr.set_ylim(-20, 40)


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
    # model = bdcn_decoder.Decoder(pretrain=None, logger=logger, in_dim=XXXXX)
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


