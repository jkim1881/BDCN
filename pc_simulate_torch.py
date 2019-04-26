import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

########## BASIC CASE
N, D_in, D_out = 1, 6, 6

# Initialize input and output
x = torch.tensor(np.ones((1,D_in)), dtype=dtype, requires_grad=True)
y = torch.tensor(np.zeros((1,D_in)), dtype=dtype, requires_grad=True)

# Initialize weights: unit basis vectors except for d2, d3
weight = torch.zeros(6, 6, dtype=dtype)
weight[0,0] = 2
weight[1,1] = 1
weight[2,2] = 0.5
weight[4,4] = -0.5

e_history = []
x_history = []
y_history = []
num_iters = 250
x_rate = 0.01
y_rate = 0.01

for iter in range(num_iters):
    print(iter)

    x_history.append(x.detach().numpy().copy())
    y_history.append(y.detach().numpy().copy())

    # Forward prediction
    pred = y.mm(weight)
    err = x - pred
    e_history.append(err.detach().numpy().copy())
    loss = err.pow(2).sum()

    # Update input/output (GD)
    loss.backward()
    with torch.no_grad():
        x -= x_rate * x.grad
        y -= y_rate * y.grad

        # Manually zero the gradients after updating weights
        x.grad.zero_()
        y.grad.zero_()

plt.subplot(3,6,1);plt.plot(range(num_iters), [yt[0,0] for yt in y_history]);plt.ylim(-1,1)
plt.subplot(3,6,2);plt.plot(range(num_iters), [yt[0,1] for yt in y_history]);plt.ylim(-1,1)
plt.subplot(3,6,3);plt.plot(range(num_iters), [yt[0,2] for yt in y_history]);plt.ylim(-1,1)
plt.subplot(3,6,4);plt.plot(range(num_iters), [yt[0,3] for yt in y_history]);plt.ylim(-1,1)
plt.subplot(3,6,5);plt.plot(range(num_iters), [yt[0,4] for yt in y_history]);plt.ylim(-1,1)
plt.subplot(3,6,6);plt.plot(range(num_iters), [yt[0,5] for yt in y_history]);plt.ylim(-1,1)

plt.subplot(3,6,7);plt.plot(range(num_iters), [et[0,0] for et in e_history]);plt.ylim(-1,1)
plt.subplot(3,6,8);plt.plot(range(num_iters), [et[0,1] for et in e_history]);plt.ylim(-1,1)
plt.subplot(3,6,9);plt.plot(range(num_iters), [et[0,2] for et in e_history]);plt.ylim(-1,1)
plt.subplot(3,6,10);plt.plot(range(num_iters), [et[0,3] for et in e_history]);plt.ylim(-1,1)
plt.subplot(3,6,11);plt.plot(range(num_iters), [et[0,4] for et in e_history]);plt.ylim(-1,1)
plt.subplot(3,6,12);plt.plot(range(num_iters), [et[0,5] for et in e_history]);plt.ylim(-1,1)

plt.subplot(3,6,13);plt.plot(range(num_iters), [xt[0,0] for xt in x_history]);plt.ylim(-1,1)
plt.subplot(3,6,14);plt.plot(range(num_iters), [xt[0,1] for xt in x_history]);plt.ylim(-1,1)
plt.subplot(3,6,15);plt.plot(range(num_iters), [xt[0,2] for xt in x_history]);plt.ylim(-1,1)
plt.subplot(3,6,16);plt.plot(range(num_iters), [xt[0,3] for xt in x_history]);plt.ylim(-1,1)
plt.subplot(3,6,17);plt.plot(range(num_iters), [xt[0,4] for xt in x_history]);plt.ylim(-1,1)
plt.subplot(3,6,18);plt.plot(range(num_iters), [xt[0,5] for xt in x_history]);plt.ylim(-1,1)

plt.show()

