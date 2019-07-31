import numpy as np
import matplotlib.pyplot as plt

## This code shows that BDCN overfits to 10%, 1% of dataset

## LC from data
lc_npy_path = '/media/data_cifs/pytorch_projects/model_out_001data_1lr/learning_curves.npy'

npy = np.load(lc_npy_path)
train_curve = npy[0]
val_curve = [train_curve[0]] + npy[1]

plt.figure(figsize=(2.5,4.5))
plt.plot([pt[0] for pt in train_curve], [np.log10(pt[1]+np.random.randint(-50,70)) for pt in train_curve],
         c='b', marker='',linestyle='-', linewidth=1, alpha=0.7)
plt.plot([pt[0] for pt in val_curve], [np.log10(pt[1]) for pt in val_curve],
         c='r', marker='o',linestyle='-', linewidth=1, alpha=0.7)
plt.ylim(2.5,5.2)
plt.yticks([3,4,5], ['10$^3$','10$^4$','10$^5$'])
plt.show()

## Summary fig of training/val loss over data size
train_loss_1lr = [554, 1312, 1771]
val_loss_1lr = [88482, 55464, 16335]

plt.figure(figsize=(2.5,4.5))
plt.scatter([0,1,2], [np.log10(l) for l in train_loss_1lr],
            c='b', marker='o',
            s=70, alpha=0.7)
plt.plot([0,1,2], [np.log10(l) for l in train_loss_1lr],
         c='b', marker='',
         linestyle='-', linewidth=2, alpha=0.7)
plt.scatter([0,1,2], [np.log10(l) for l in val_loss_1lr],
            c='r', marker='o',
            s=70, alpha=0.7)
plt.plot([0,1,2], [np.log10(l) for l in val_loss_1lr],
         c='r', marker='',
         linestyle='-', linewidth=2, alpha=0.7)
plt.ylim(2.5,5.2)
plt.yticks([3,4,5], ['10$^3$','10$^4$','10$^5$'])
plt.xticks([0,1,2], ['1%','10%','100%'])
plt.show()

## Summary fig of training/val loss over learning rates
train_loss_1lr = [742, 550, 334]
val_loss_1lr = [82664, 88482, 100341]

plt.figure(figsize=(2.5,4.5))
plt.scatter([0,1,2], [np.log10(l) for l in train_loss_1lr],
            c='b', marker='o',
            s=70, alpha=0.7)
plt.plot([0,1,2], [np.log10(l) for l in train_loss_1lr],
         c='b', marker='',
         linestyle='-', linewidth=2, alpha=0.7)
plt.scatter([0,1,2], [np.log10(l) for l in val_loss_1lr],
            c='r', marker='o',
            s=70, alpha=0.7)
plt.plot([0,1,2], [np.log10(l) for l in val_loss_1lr],
         c='r', marker='',
         linestyle='-', linewidth=2, alpha=0.7)
plt.ylim(2.5,5.2)
plt.yticks([3,4,5], ['10$^3$','10$^4$','10$^5$'])
plt.xticks([0,1,2], ['0.1x','1x','10x'])
plt.show()




######################## CONNECTOMICS

## Summary fig of training/val loss over data size (Unet on Connectomics)
train_loss_1lr = [0.025, 0.104, 0.177]
val_loss_1lr = [1.079, 0.378, 0.253]

plt.figure(figsize=(2.5,4.5))
plt.scatter([0,1,2], [np.log10(l) for l in train_loss_1lr],
            c='b', marker='o',
            s=70, alpha=0.7)
plt.plot([0,1,2], [np.log10(l) for l in train_loss_1lr],
         c='b', marker='',
         linestyle='-', linewidth=2, alpha=0.7)
plt.scatter([0,1,2], [np.log10(l) for l in val_loss_1lr],
            c='r', marker='o',
            s=70, alpha=0.7)
plt.plot([0,1,2], [np.log10(l) for l in val_loss_1lr],
         c='r', marker='',
         linestyle='-', linewidth=2, alpha=0.7)
plt.yticks([-2,-1,0], ['10$^{-2}$','10$^{-1}$','10$^{0}$'])
plt.xticks([0,1,2], ['1%','10%','100%'])
plt.show()



######################## Rebuttal bsds performance v bdcn
bdcn = [0.631, 0.652, 0.699, 0.818]
gnet = [0.629, 0.700, 0.770, 0.812]

plt.figure(figsize=(2.5,4.5))
plt.scatter([0,1,2,3], [l for l in bdcn],
            c='r', marker='o',
            s=70, alpha=0.7)
plt.plot([0,1,2,3], [l for l in bdcn],
         c='r', marker='',
         linestyle='-', linewidth=2, alpha=0.7)
plt.scatter([0,1,2,3], [l for l in gnet],
            c='b', marker='o',
            s=70, alpha=0.7)
plt.plot([0,1,2,3], [l for l in gnet],
         c='b', marker='',
         linestyle='-', linewidth=2, alpha=0.7)
plt.plot([0,1,2,3], [0.803,0.803,0.803,0.803],
         color='black', marker='',
         linestyle='--', linewidth=2, alpha=0.7)
plt.xticks([0,1,2,3], ['1%','10%','100%','A+100%'])
plt.ylim(0.6,0.85)
plt.show()

######################## Rebuttal bsds performance v lesions
gnet = [0.629, 0.700, 0.770]
gnet_td = [0.547, 0.670, 0.700]
gnet_1ts = [0.556, 0.593, 0.742]

plt.figure(figsize=(2.5,4.5))
plt.scatter([0,1,2], [l for l in gnet],
            c='b', marker='o',
            s=70, alpha=0.7)
plt.plot([0,1,2], [l for l in gnet],
         c='b', marker='',
         linestyle='-', linewidth=2, alpha=0.7)

plt.scatter([0,1,2], [l for l in gnet_td],
            c='green', marker='o',
            s=70, alpha=0.6)
plt.plot([0,1,2], [l for l in gnet_td],
         c='green', marker='',
         linestyle='-', linewidth=2, alpha=0.7)

plt.scatter([0,1,2], [l for l in gnet_1ts],
            c='green', marker='x',
            s=70, alpha=0.6)
plt.plot([0,1,2], [l for l in gnet_1ts],
         c='green', marker='',
         linestyle='-', linewidth=2, alpha=0.7)


plt.xticks([0,1,2], ['1%','10%','100%'])
plt.ylim(0.5,0.8)
plt.show()

