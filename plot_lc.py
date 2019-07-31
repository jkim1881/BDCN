import numpy as np
import matplotlib.pyplot as plt

lc_npy_path = '/media/data_cifs/pytorch_projects/model_out_001data_1lr/learning_curves.npy'

npy = np.load(lc_npy_path)
train_curve = npy[0]
val_curve = npy[1]

plt.plot([pt[0] for pt in train_curve],[pt[1] for pt in train_curve])
plt.show()
