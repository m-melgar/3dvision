import numpy as np
import scipy as scipy
from scipy import linalg as LA


def factorize_camera_matrix(p: np.ndarray) -> tuple(np.ndarray, np.ndarray, np.ndarray):
    """
    Factorizes camera parameter P as K internal parameters, R external parameters and c camera center
    :param p: np.ndarray([n.m])
            camera matrix P
    :return: np.ndarray([n.m]), np.ndarray([n.m]), np.ndarray([n.m])
            K, R, c
    """
    m = p[:, 0:3]

    k, r = LA.rq(m)

    t = np.diag(np.sign(np.diag(k)))

    if scipy.linalg.det(t) < 0:
        t[1, 1] *= -1

    k = np.dot(k, t)
    r = np.dot(t, r)

    c = np.dot(LA.inv(-m), p[:, 3])
    return k, r, c
