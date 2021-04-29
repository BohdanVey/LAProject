import numpy as np
import torch
from sklearn.metrics.pairwise import pairwise_distances
import torch.nn as nn
from hausdorf_loss_internet import AveragedHausdorffLoss


def square_loss(a, b):
    c = a - b
    d = np.sum(c ** 2)
    return d / a.shape[0] / a.shape[1]


def dice_loss(a, b):
    assert len(a) == len(b) and len(a[0]) == len(b[0])
    numerator = np.sum(2 * a * b)
    denominator = np.sum(a * a + b * b)
    return 1 - numerator / denominator


def hausdorff_loss(a, b, max_ahd=10 ** 9):
    if len(a) == 0 or len(b) == 0:
        return max_ahd
    d2_matrix = pairwise_distances(a, b, metric='euclidean')

    res = np.average(np.min(d2_matrix, axis=0)) + \
          np.average(np.min(d2_matrix, axis=1))

    return res


a = np.random.randint(-100, 100, (800, 800))
b = np.random.randint(-100, 100, (800, 800))
print("Square Loss of different matrix: ", square_loss(a, b))
print("Square Loss of same matrix: ", square_loss(a, a))
print("Square loss torch: ", nn.MSELoss()(torch.tensor(a, dtype=torch.double), torch.tensor(b, dtype=torch.double)))
print("Dice Loss of different matrix: ", dice_loss(a, b))
print("Dice Loss of same matrix: ", dice_loss(a, a))
print("Dice loss torch: ", nn.MSELoss()(torch.tensor(a, dtype=torch.double), torch.tensor(b, dtype=torch.double)))

print("Hausdorff Loss of different matrix: ", hausdorff_loss(a, b))
print("Hausdorff Loss of same matrix: ", hausdorff_loss(a, a))
print("Hausdorff loss from internet",
      AveragedHausdorffLoss().forward(torch.tensor(a, dtype=torch.double), torch.tensor(b, dtype=torch.double)))
