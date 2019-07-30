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


def main():
    import matplotlib.pyplot as plt

    ### ASSUME OUTPUT FILE AND META ARE CO-ALIGNED
    center_gt = []
    surround_gt = []
    predictions = []

    out_data = np.load('/Users/junkyungkim/Desktop/BSDS_vgg_gratings_simple_gratings_test_2019_06_04_20_38_56_725053.npz')
    out_data_arr = out_data['test_dict'].copy()
    meta_arr = np.reshape(np.load('/Users/junkyungkim/Desktop/1.npy'), [-1, 11])

    f = plt.figure(figsize=(4,4))
    axarr = f.subplots(4,4) #(4, 4)
    for ir, rmin in enumerate([40,60,80,100]):
        for ith, thetamin in enumerate([-22.5, 22.5, 67.5, 112.5]):
            center_gt = []
            surround_gt = []
            predictions = []
            cond = screen(meta_arr[:,3].astype(np.float), meta_arr[:,5].astype(np.float), meta_arr[:,4].astype(np.float),
                          r1min=rmin, r1max=rmin+20, lambda1min=None, lambda1max=None, thetamin=thetamin, thetamax=thetamin+45)
            for i in xrange(out_data_arr.size):

                out_deg = ((np.arctan2(out_data_arr[i]['logits'][0, 0], out_data_arr[i]['logits'][0, 1])) * 180 / np.pi) % 180
                label_deg = ((np.arctan2(out_data_arr[i]['labels'][0, 0], out_data_arr[i]['labels'][0, 1])) * 180 / np.pi) % 180

                # r1, r2 ~ [40, 120], r2 = 2*r1
                # lambda1, lambda2~ [30 90]
                #
                # [im_sub_path, im_fn, iimg, r1, theta1, lambda1, shift1, r2, theta2, lambda2, shift2]

                if cond[i]:
                    center_gt.append(meta_arr[i, 4].astype(np.float))
                    surround_gt.append(meta_arr[i, 8].astype(np.float))
                    predictions.append(out_deg)

            if len(center_gt) > 0:
                # # plot
                # print('filtered ' + str(len(predictions)) + ' data')
                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(16, 4))
                # plt.subplot(141)
                # plt.scatter(center_gt, np.array(predictions), s=10, vmin=0, vmax=180)
                #
                import numpy.polynomial.polynomial as poly
                # plt.subplot(142)
                cs_diff = orientation_diff(center_gt, surround_gt) #center - surround in x axis
                out_gt_diff = orientation_diff(predictions, center_gt) #pred - gt in y axis
                cs_diff_collapsed, out_gt_diff_collapsed = collapse_points(cs_diff, out_gt_diff)
                coefs = poly.polyfit(cs_diff_collapsed, out_gt_diff_collapsed, 5)
                ffit = poly.polyval(np.arange(-90, 90, 1), coefs)
                axarr[ir,ith].scatter(cs_diff_collapsed, out_gt_diff_collapsed, s=40, alpha=0.25, vmin=0, vmax=180)
                # coefs = poly.polyfit(cs_diff, out_gt_diff, 5)
                # ffit = poly.polyval(np.arange(-90, 90, 1), coefs)
                # axarr[ir, ith].scatter(cs_diff, out_gt_diff, s=15, alpha=0.3, vmin=0, vmax=180)
                axarr[ir,ith].plot(np.arange(-90, 90, 1), ffit, linewidth=3, alpha=0.5, color='black')
                axarr[ir,ith].plot(np.arange(-90, 90, 1), [0] * np.arange(-90, 90, 1).size, color='black')
                axarr[ir,ith].set_xlim(0, 87)
                axarr[ir,ith].set_ylim(-20, 40)
                axarr[ir,ith].set_title('r in ' + str([rmin, rmin+20]) + ', tht in ' + str([thetamin, thetamin+45]))


                import numpy.polynomial.polynomial as poly
                ff = plt.figure(figsize=(4, 4))
                axr = ff.subplots(1, 1)  # (4, 4)
                cs_diff = [0,10,20,30,40,50,60,70,80,90]
                # HUMAN DATA
                # out_gt_diff = [-0.18281291942873512,
                #                1.8150305059760774,
                #                2.6838843821231793,
                #                1.997825226463081,
                #                0.48926489924610905,
                #                -0.3967642281601318,
                #                -0.3215298030419529,
                #                -0.6945125640943974,
                #                -0.5774387719354115,
                #                -0.4699194241854827] 
                cs_diff_collapsed, out_gt_diff_collapsed = collapse_points(cs_diff, out_gt_diff)
                coefs = poly.polyfit(cs_diff_collapsed, out_gt_diff_collapsed, 5)
                ffit = poly.polyval(np.arange(-90, 90, 1), coefs)
                axr.scatter(cs_diff_collapsed, out_gt_diff_collapsed, s=40, alpha=0.45, vmin=0, vmax=180)
                # coefs = poly.polyfit(cs_diff, out_gt_diff, 5)
                # ffit = poly.polyval(np.arange(-90, 90, 1), coefs)
                # axarr[ir, ith].scatter(cs_diff, out_gt_diff, s=15, alpha=0.3, vmin=0, vmax=180)
                axr.plot(np.arange(-90, 90, 1), ffit, linewidth=3, alpha=0.5, color='black')
                axr.plot(np.arange(-90, 90, 1), [0] * np.arange(-90, 90, 1).size, color='black')
                axr.set_xlim(0, 87)
                axr.set_ylim(-2, 4)
                ff.show()
                #
                # plt.subplot(143)
                # x_list, y_mu, y_25, y_75 = cluster_points(cs_diff, out_gt_diff, 10)
                # plt.scatter(cs_diff, out_gt_diff, s=25, alpha=0.1, vmin=0, vmax=180, color='black')
                # plt.plot(np.arange(-90, 90, 1), [0]*np.arange(-90, 90, 1).size, color='black')
                # plt.fill_between(x_list, y_25, y_75, alpha=0.5)
                # plt.plot(x_list, y_mu, linewidth=3, alpha=0.5, color='red')
                # plt.xlim(-90, 90)
                # plt.ylim(-25, 25)
                #
                # plt.subplot(144)
                # cs_diff_collapsed, out_gt_diff_collapsed = collapse_points(cs_diff, out_gt_diff)
                # x_list, y_mu, y_25, y_75 = cluster_points(cs_diff_collapsed, out_gt_diff_collapsed, 10)
                # plt.scatter(cs_diff_collapsed, out_gt_diff_collapsed, s=25, alpha=0.1, vmin=0, vmax=180, color='black')
                # plt.plot(np.arange(-90, 90, 1), [0]*np.arange(-90, 90, 1).size, color='black')
                # plt.fill_between(x_list, y_25, y_75, alpha=0.5)
                # plt.plot(x_list, y_mu, linewidth=3, alpha=0.5, color='red')
                # plt.xlim(0, 90)
                # plt.ylim(-20, 50)
                #
                # plt.show()

    plt.show()

if __name__ == '__main__':
    main()


