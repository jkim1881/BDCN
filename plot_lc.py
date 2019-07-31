import numpy as np
import matplotlib.pyplot as plt

lc_npy_path = '/media/data_cifs/pytorch_projects/model_out_001data_1lr/learning_curves.npy'

npy = np.load(lc_npy_path)
train_curve = npy[0]
val_curve = [train_curve[0]] + npy[1]

plt.figure(figsize=(1.5,3))
plt.plot([pt[0] for pt in train_curve], [np.log10(pt[1]) for pt in train_curve])
plt.plot([pt[0] for pt in val_curve], [np.log10(pt[1]) for pt in val_curve])
plt.show()
