import numpy as np
from numpy.linalg import norm
from random import normalvariate
from math import sqrt


def random_unit_vector(n):
    """
    Generates random unit vector
    :param n: vector length
    :return: unit vector of length n
    """
    denormalized = [normalvariate(0, 1) for _ in range(n)]
    the_norm = sqrt(sum(x * x for x in denormalized))
    return [x / the_norm for x in denormalized]


def power_step(A, A_curr, svd_rez, error=1e-7):
    """
    Approximates greatest singular value of matrix A_curr and left/right singular vectors which correspond to it
    :param A: initial matrix
    :param A_curr: input matrix with subtracted already determined sigma, v, u
    :param svd_rez: dict with svd results
    :param error: required maximal angle between v_curr and v_prev
    """
    n, m = A_curr.shape
    k = min(n, m)
    # v = random_unit_vector(k)л
    v = np.ones(k) / sqrt(k)
    if n > m:
        A_curr = A_curr.T @ A_curr
    elif n < m:
        A_curr = A_curr @ A_curr.T

    while True:
        print()
        Av = A_curr @ v
        v_new = Av / np.linalg.norm(Av)
        if np.abs(np.dot(v, v_new)) > 1 - error:
            break
        v = v_new

    if n > m:
        u_denormalized = A @ v_new
        sigma = np.linalg.norm(u_denormalized)
        svd_rez["s"].append(sigma)
        svd_rez["u"].append(u_denormalized / sigma)
        svd_rez["v"].append(v_new)
    else:
        v_denormalized = A.T @ v_new
        sigma = np.linalg.norm(v_denormalized)
        svd_rez["s"].append(sigma)
        svd_rez["u"].append(v_new)
        svd_rez["v"].append(v_denormalized / sigma)


def svd(A, _rank):
    """
    Approximates SVD of matrix A
    :param A: matrix to be decomposed
    :param _rank: number of greatest singular values to be approximated
    :return: te result of svd
    """
    n, m = A.shape
    svd_rez = {
        "u": [],
        "s": [],
        "v": []
    }

    A_reserve = A.copy()
    for i in range(_rank):
        print(f"step - {i}")
        power_step(A, A_reserve, svd_rez)
        A_reserve -= svd_rez["s"][-1] * np.outer(svd_rez["u"][-1], svd_rez["v"][-1])

    return np.array(svd_rez["s"]), np.array(svd_rez["u"]).T, np.array(svd_rez["v"])


if __name__ == "__main__":
    movieRatings = np.array([
        [2, 5, 3],
        [1, 2, 1],
        [4, 1, 1],
        [3, 5, 2],
        [5, 3, 1],
        [4, 5, 5],
        [2, 4, 2],
        [2, 2, 5],
    ], dtype='float64')

    values, left_s, rigth_s = svd(movieRatings, min(movieRatings.shape))
    print(left_s)
    print(values)
    print(rigth_s)
    print(np.matmul(np.matmul(left_s, np.diag(values)), rigth_s))

    print("\n\n\n")
    u, s, v = np.linalg.svd(movieRatings, full_matrices=False)
    print(u)
    print(s)
    print(v)
    print(np.matmul(np.matmul(u, np.diag(s)), v))
