import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


y_train = np.loadtxt("data/processed/y_train.txt", dtype=int)
l0_l1 = [y_train[i, 0]*y_train[i, 1] for i in range(y_train.shape[0])]
l1_l2 = [y_train[i, 1]*y_train[i, 2] for i in range(y_train.shape[0])]
l0_l3 = [y_train[i, 0]*y_train[i, 3] for i in range(y_train.shape[0])]
l0_l4 = [y_train[i, 0]*y_train[i, 4] for i in range(y_train.shape[0])]
l0_l5 = [y_train[i, 0]*y_train[i, 5] for i in range(y_train.shape[0])]
distribution = []
edge_distribution = [np.sum(l0_l1), np.sum(l1_l2), np.sum(l0_l3), np.sum(l0_l4), np.sum(l0_l5)]
for i in range(y_train.shape[0]):
    distribution.append(np.sum(y_train[i, :]))

num_list = [0 for i in range(15)]
for value in distribution:
    if value == 1:
        num_list[0] += 1
    elif value == 2:
        num_list[1] += 1
    elif value == 3:
        num_list[2] += 1
    elif value == 4:
        num_list[3] += 1
    elif value == 5:
        num_list[4] += 1
    elif value == 6:
        num_list[5] += 1

# X = [1, 2, 3, 4, 5, 6]
# Y = num_list
# fig = plt.figure()
# plt.bar(X, Y, 0.4, color="black")
# plt.xlabel("label num")
# plt.ylabel("train data num")
# plt.title("")

X = ["l0_l1", "l1_l2", "l0_l3", "l0_l4", "l0_l5"]
Y = edge_distribution
fig = plt.figure()
plt.bar(X, Y, 0.4, color="black")
plt.xlabel("edge")
plt.ylabel("num")
plt.title("")

plt.show()

