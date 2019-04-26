import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

########## STANDARD CASE
# random image
input =  np.ones((5,1))#np.random.rand(5,1)
output = np.zeros((3,1))

# unit basis vectors except for d2, d3
weight = np.zeros((3,5))
weight[0,0] = 2
weight[1,3] = 0.5
weight[2,4] = 1

error = []
x = []
y = []
num_iters = 250
ff_gate = 0.01
fb_switch = 0.01

for iter in range(num_iters):
    # output = np.zeros((3,1))
    # output[2] = 0.7
    # output[0] -= 0.005
    # output[1] -= 0.005
    # output[2] += 0.005

    x.append(input.copy())
    y.append(output.copy())
    pred = np.dot(np.transpose(weight), output)
    err = input-pred
    error.append(err)

    output += ff_gate*(np.dot(weight,err))
    input = input*(1-fb_switch) + pred*fb_switch

plt.subplot(3,5,1);plt.plot(range(num_iters), [yt[0] for yt in y]);plt.ylim(0,1)
plt.subplot(3,5,4);plt.plot(range(num_iters), [yt[1] for yt in y]);plt.ylim(0,1)
plt.subplot(3,5,5);plt.plot(range(num_iters), [yt[2] for yt in y]);plt.ylim(0,1)

plt.subplot(3,5,6);plt.plot(range(num_iters), [et[0] for et in error]);plt.ylim(0,1)
plt.subplot(3,5,7);plt.plot(range(num_iters), [et[1] for et in error]);plt.ylim(0,1)
plt.subplot(3,5,8);plt.plot(range(num_iters), [et[2] for et in error]);plt.ylim(0,1)
plt.subplot(3,5,9);plt.plot(range(num_iters), [et[3] for et in error]);plt.ylim(0,1)
plt.subplot(3,5,10);plt.plot(range(num_iters), [et[4] for et in error]);plt.ylim(0,1)

plt.subplot(3,5,11);plt.plot(range(num_iters), [xt[0] for xt in x]);plt.ylim(0,1)
plt.subplot(3,5,12);plt.plot(range(num_iters), [xt[1] for xt in x]);plt.ylim(0,1)
plt.subplot(3,5,13);plt.plot(range(num_iters), [xt[2] for xt in x]);plt.ylim(0,1)
plt.subplot(3,5,14);plt.plot(range(num_iters), [xt[3] for xt in x]);plt.ylim(0,1)
plt.subplot(3,5,15);plt.plot(range(num_iters), [xt[4] for xt in x]);plt.ylim(0,1)

plt.show()


########## STANDARD CASE + FB is a derivateive
# random image
input =  np.ones((5,1)) # np.random.rand(5,1)
output = np.zeros((3,1))

# unit basis vectors except for d2, d3
weight = np.zeros((3,5))
weight[0,0] = 2
weight[1,3] = 0.5
weight[2,4] = 1

error = []
x = []
y = []
num_iters = 250
ff_gate = 0.01
fb_switch = 0.01

for iter in range(num_iters):
    # output = np.zeros((3,1))
    # output[2] = 0.7
    # output[0] -= 0.005
    # output[1] -= 0.005
    # output[2] += 0.005

    x.append(input.copy())
    y.append(output.copy())
    pred = np.dot(weight, input)
    err = output-pred
    error.append(err)

    input += ff_gate*(np.dot(np.transpose(weight), err))
    output = output*(1-fb_switch) + pred*fb_switch


plt.subplot(3,5,1);plt.plot(range(num_iters), [yt[0] for yt in y]);plt.ylim(0,1)
plt.subplot(3,5,4);plt.plot(range(num_iters), [yt[1] for yt in y]);plt.ylim(0,1)
plt.subplot(3,5,5);plt.plot(range(num_iters), [yt[2] for yt in y]);plt.ylim(0,1)

plt.subplot(3,5,6);plt.plot(range(num_iters), [et[0] for et in error]);plt.ylim(0,1)
plt.subplot(3,5,9);plt.plot(range(num_iters), [et[1] for et in error]);plt.ylim(0,1)
plt.subplot(3,5,10);plt.plot(range(num_iters), [et[2] for et in error]);plt.ylim(0,1)

plt.subplot(3,5,11);plt.plot(range(num_iters), [xt[0] for xt in x]);plt.ylim(0,1)
plt.subplot(3,5,12);plt.plot(range(num_iters), [xt[1] for xt in x]);plt.ylim(0,1)
plt.subplot(3,5,13);plt.plot(range(num_iters), [xt[2] for xt in x]);plt.ylim(0,1)
plt.subplot(3,5,14);plt.plot(range(num_iters), [xt[3] for xt in x]);plt.ylim(0,1)
plt.subplot(3,5,15);plt.plot(range(num_iters), [xt[4] for xt in x]);plt.ylim(0,1)

plt.show()


########## CASE WITH MAX POOLING (second output unit pools from 2,3,4-th units)
# random image
input =  np.zeros((5,1))
input[0] = 1
input[4] = 1
input[2] = 0.5
input[3] = 1     #[1, 0, 0.5, 1, 1]
output = np.zeros((3,1))

# 5x5 identity matrix
weight = np.identity(5)

error = []
x = []
y = []
num_iters = 250
ff_gate = 0.01
fb_switch = 0.01

for iter in range(num_iters):
    # output = np.zeros((3,1))
    # output[2] = 0.7
    # output[0] -= 0.005
    # output[1] -= 0.005
    # output[2] += 0.005

    # FB is a gradient wrt FF
    x.append(input.copy())
    y.append(output.copy())
    pred = np.dot(np.transpose(weight), np.array([output[0], output[1], output[1], output[1], output[2]]))
    err = input-pred
    error.append(err)

    pre_output = np.dot(weight, err)
    output[0] += ff_gate*pre_output[0]
    output[1] += ff_gate*np.max([pre_output[1], pre_output[2], pre_output[3]])
    output[2] += ff_gate*pre_output[4]
    input = input*(1-fb_switch) + pred*fb_switch

plt.subplot(3,5,1);plt.plot(range(num_iters), [yt[0] for yt in y]);plt.ylim(0,1)
plt.subplot(3,5,4);plt.plot(range(num_iters), [yt[1] for yt in y]);plt.ylim(0,1)
plt.subplot(3,5,5);plt.plot(range(num_iters), [yt[2] for yt in y]);plt.ylim(0,1)

plt.subplot(3,5,6);plt.plot(range(num_iters), [et[0] for et in error]);plt.ylim(0,1)
plt.subplot(3,5,7);plt.plot(range(num_iters), [et[1] for et in error]);plt.ylim(0,1)
plt.subplot(3,5,8);plt.plot(range(num_iters), [et[2] for et in error]);plt.ylim(0,1)
plt.subplot(3,5,9);plt.plot(range(num_iters), [et[3] for et in error]);plt.ylim(0,1)
plt.subplot(3,5,10);plt.plot(range(num_iters), [et[4] for et in error]);plt.ylim(0,1)

plt.subplot(3,5,11);plt.plot(range(num_iters), [xt[0] for xt in x]);plt.ylim(0,1)
plt.subplot(3,5,12);plt.plot(range(num_iters), [xt[1] for xt in x]);plt.ylim(0,1)
plt.subplot(3,5,13);plt.plot(range(num_iters), [xt[2] for xt in x]);plt.ylim(0,1)
plt.subplot(3,5,14);plt.plot(range(num_iters), [xt[3] for xt in x]);plt.ylim(0,1)
plt.subplot(3,5,15);plt.plot(range(num_iters), [xt[4] for xt in x]);plt.ylim(0,1)

plt.show()