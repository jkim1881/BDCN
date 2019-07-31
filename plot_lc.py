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
train_loss_1lr = [554, 1312, 1771]
val_loss_1lr = [88482, 55464, 16335]

plt.figure(figsize=(3,3))
plt.scatter([0,1,2], [np.log10(l) for l in train_loss_1lr],
            c='b', marker='o',
            s=40, alpha=0.7)
plt.plot([0,1,2], [np.log10(l) for l in train_loss_1lr],
         c='b', marker='',
         linestyle='-', linewidth=1.4, alpha=0.7)
plt.scatter([0,1,2], [np.log10(l) for l in val_loss_1lr],
            c='r', marker='o',
            s=40, alpha=0.7)
plt.plot([0,1,2], [np.log10(l) for l in val_loss_1lr],
         c='r', marker='',
         linestyle='-', linewidth=1.4, alpha=0.7)
plt.yticks([3,4,5], ['10$^3','10$^4','10$^5'])
plt.xticks([0,1,2], ['1%','10%','100%'])
plt.show()