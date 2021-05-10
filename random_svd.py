import numpy as np
from svd import svd
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread


def rSVD(A, rank):
    """
    Implementation of randomized SVD
    :param A: input matrix
    :param rank: rank of SVD decomposition
    :return: SVD decomposition matrices
    """
    n, m = A.shape
    P = np.random.randn(m, rank)
    Z = A @ P
    q, r = np.linalg.qr(Z, mode="reduced")
    Y = q.T @ A
    s, uy, v = svd(Y, min(min(Y.shape), rank))
    u = q @ uy
    return s, u, v


if __name__ == "__main__":
    # A = np.array([
    #     [2, 5, 3],
    #     [1, 2, 1],
    #     [4, 1, 1],
    #     [3, 5, 2],
    #     [5, 3, 1],
    #     [4, 5, 5],
    #     [2, 4, 2],
    #     [2, 2, 5],
    # ], dtype='float64')
    A = imread("Apollo-11-Crew-Photo1.jpg")
    X = np.mean(A, axis=2)

    u, s, vt = np.linalg.svd(X, full_matrices=0)

    r = 400
    q = 1
    p = 5

    rs, ru, rvt = rSVD(X, r)

    XSVD = u[:, :(r + 1)] @ np.diag(s[:(r + 1)]) @ vt[:(r + 1), :]
    XrSVD = ru[:, :(r + 1)] @ np.diag(rs[:(r + 1)]) @ rvt[:(r + 1), :]

    fig, axs = plt.subplots(1, 3)

    plt.set_cmap("gray")
    axs[0].imshow(256 - X)
    axs[0].axis("off")
    axs[1].imshow(256 - XSVD)
    axs[1].axis("off")
    axs[2].imshow(256 - XrSVD)
    axs[2].axis("off")

    plt.show()
    #
    # values, left_s, rigth_s = rSVD(A, min(A.shape), 1, 0)
    # print(left_s)
    # print(values)
    # print(rigth_s)

    # print(np.matmul(np.matmul(left_s, np.diag(values)), rigth_s))
    #
    # print("\n\n\n")
    # u, s, v = np.linalg.svd(movieRatings, full_matrices=False)
    # print(u)
    # print(s)
    # print(v)
    # print(np.matmul(np.matmul(u, np.diag(s)), v))
