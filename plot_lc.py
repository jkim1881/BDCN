import numpy as np
import matplotlib.pyplot as plt

## LC from data
# lc_npy_path = '/media/data_cifs/pytorch_projects/model_out_001data_01lr/learning_curves.npy'
#
# npy = np.load(lc_npy_path)
# train_curve = npy[0]
# val_curve = [train_curve[0]] + npy[1]
#
# plt.figure(figsize=(3,3))
# plt.plot([pt[0] for pt in train_curve], [pt[1] for pt in train_curve])
# plt.plot([pt[0] for pt in val_curve], [pt[1] for pt in val_curve])
# plt.show()

## Summary fig of training/val loss over conditions
train_loss_1lr = [1771, 1312, 554]
val_loss_1lr = [16335, 55464, 88482]
plt.scatter([0,1,2], train_loss_1lr,
            c='b', marker='o',
            s=40, alpha=0.7)
plt.plot([0,1,2], train_loss_1lr,
         c='b', marker='',
         linestyle='-', linewidth=1.4, alpha=0.7)
plt.scatter([0,1,2], val_loss_1lr,
            c='r', marker='o',
            s=40, alpha=0.7)
plt.plot([0,1,2], val_loss_1lr,
         c='r', marker='',
         linestyle='-', linewidth=1.4, alpha=0.7)