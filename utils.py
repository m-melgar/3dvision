import numpy as np
import scipy as scipy
from scipy import linalg as LA


def factorize_camera_matrix(P: np.ndarray) -> tuple(np.ndarray, np.ndarray, np.ndarray):
    """
    Factorizes camera parameter P as K internal parameters, R external parameters and C camera center
    :param P: np.ndarray([n.m])
            camera matrix P
    :return: np.ndarray([n.m]), np.ndarray([n.m]), np.ndarray([n.m])
            K, R, C
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
