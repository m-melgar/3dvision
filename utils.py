import numpy as np
import scipy as scipy
from scipy import linalg as LA


def factorize_camera_matrix(P):
    """

    :param P: camera matrix P
    :return: K internal parameters, R external parameters, C camera center
    """
    M = P[:, 0:3]

    K, R = LA.rq(M)

    T = np.diag(np.sign(np.diag(K)))

    if scipy.linalg.det(T) < 0:
        T[1, 1] *= -1

    K = np.dot(K, T)
    R = np.dot(T, R)

    C = np.dot(LA.inv(-M), P[:, 3])
    return K, R, C
