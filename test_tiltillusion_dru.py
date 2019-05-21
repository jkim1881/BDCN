import numpy as np
import re
import os
import sys
import cv2


def orientation_diff(array1, array2):
    concat = np.concatenate((np.expand_dims(array1, axis=1),
                             np.expand_dims(array2, axis=1)), axis=1)
    diffs = np.concatenate((np.expand_dims(concat[:,0] - concat[:,1], axis=1),
                            np.expand_dims(concat[:,0] - concat[:,1] - 180, axis=1),
                            np.expand_dims(concat[:,0] - concat[:,1] + 180, axis=1)), axis=1)
    diffs_argmin = np.argmin(np.abs(diffs), axis=1)
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


def main():
    ### ASSUME OUTPUT FILE AND META ARE CO-ALIGNED
    center_gt = []
    surround_gt = []
    predictions = []

    out_data = np.load('/Users/junkyungkim/Desktop/BSDS_vgg_gratings_simple_gratings_test_2019_05_20_15_07_49_616727.npz')
    out_data_arr = out_data['test_dict'].copy()
    meta_arr = np.reshape(np.load('/Users/junkyungkim/Desktop/1.npy'), [-1, 11])

    for i in xrange(out_data_arr.size):
        print(str(i))
        filenum = int(os.path.split(out_data_arr[i]['image_paths'][0])[1].split('_')[1].split('.')[0])

        if i != filenum:
            raise ValueError('file idx and output idx do not match')

        out_deg = ((np.arctan2(out_data_arr[i]['logits'][0, 0], out_data_arr[i]['logits'][0, 1])) * 180 / np.pi) % 180
        label_deg = ((np.arctan2(out_data_arr[i]['labels'][0, 0], out_data_arr[i]['labels'][0, 1])) * 180 / np.pi) % 180

        if label_deg - float(meta_arr[i, 4]) > 0.001:
            print('out label: '+str(label_deg))
            print('meta label: '+str(meta_arr[i,1]))
            raise ValueError('output label and meta label do not match')
            import ipdb;ipdb.set_trace()

        # [im_sub_path, im_fn, iimg, r1, theta1, lambda1, shift1, r2, theta2, lambda2, shift2]
        # if float(meta_arr[i, 3]) < float(meta_arr[i, 5])*1.5: #radius smaller than 1.5x lambda
        # if float(meta_arr[i, 5]) > 50:  # lambda-based filtering
        if float(meta_arr[i, 4]) < 100 or float(meta_arr[i, 4]) > 80:  # lambda-based filtering
            center_gt.append(meta_arr[i, 4].astype(np.float))
            surround_gt.append(meta_arr[i, 8].astype(np.float))
            predictions.append(out_deg)

    # plot
    print('filtered ' + str(len(predictions)) + ' data')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 4))
    plt.subplot(141)
    plt.scatter(center_gt, np.array(predictions), s=10, vmin=0, vmax=180)

    import numpy.polynomial.polynomial as poly
    plt.subplot(142)
    cs_diff = orientation_diff(center_gt, surround_gt) #center - surround in x axis
    out_gt_diff = orientation_diff(predictions, center_gt) #pred - gt in y axis
    coefs = poly.polyfit(cs_diff, out_gt_diff, 5)
    ffit = poly.polyval(np.arange(-90, 90, 1), coefs)
    plt.scatter(cs_diff, out_gt_diff, s=15, alpha=0.3, vmin=0, vmax=180)
    plt.plot(np.arange(-90, 90, 1), ffit, linewidth=3, alpha=0.5, color='black')
    plt.xlim(0, 90)
    plt.ylim(-60, 60)

    plt.subplot(143)
    x_list, y_mu, y_25, y_75 = cluster_points(cs_diff, out_gt_diff, 10)
    plt.scatter(cs_diff, out_gt_diff, s=25, alpha=0.1, vmin=0, vmax=180, color='black')
    plt.plot(np.arange(-90, 90, 1), [0]*np.arange(-90, 90, 1).size, color='black')
    plt.fill_between(x_list, y_25, y_75, alpha=0.5)
    plt.plot(x_list, y_mu, linewidth=3, alpha=0.5, color='red')
    plt.xlim(-90, 90)
    plt.ylim(-25, 25)

    plt.subplot(144)
    cs_diff_collapsed, out_gt_diff_collapsed = collapse_points(cs_diff, out_gt_diff)
    x_list, y_mu, y_25, y_75 = cluster_points(cs_diff_collapsed, out_gt_diff_collapsed, 10)
    plt.scatter(cs_diff_collapsed, out_gt_diff_collapsed, s=25, alpha=0.1, vmin=0, vmax=180, color='black')
    plt.plot(np.arange(-90, 90, 1), [0]*np.arange(-90, 90, 1).size, color='black')
    plt.fill_between(x_list, y_25, y_75, alpha=0.5)
    plt.plot(x_list, y_mu, linewidth=3, alpha=0.5, color='red')
    plt.xlim(0, 90)
    plt.ylim(-20, 50)

    plt.show()


if __name__ == '__main__':
    main()


