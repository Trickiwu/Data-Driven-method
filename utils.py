import numpy as np
from scipy import sparse as ssparse
from scipy.sparse import linalg
from scipy import linalg

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from pypower import api


def left_solve(A, B):
    if ssparse.issparse(A) or ssparse.issparse(B):
        return ssparse.linalg.spsolve(A, B)
    return linalg.solve(A, B)


def right_solve(B, A):
    if ssparse.issparse(A) or ssparse.issparse(B):
        return ssparse.linalg.spsolve(A.T, B.T).T
    return linalg.solve(A.T, B.T).T


def load_case(casename):
    assert '.' not in casename

    if casename == 'case9':
        return api.case9()
    if casename == 'case14':
        return api.case14()
    elif casename == 'case30':
        return api.case30()

    raise NotImplementedError(f'casename={casename} not implemented')


def recursive_visualize(name, d, indent=0):
    if isinstance(d, dict):
        print('\t'*indent, 'name:', name)
        for k, v in d.items():
            recursive_visualize(k, v, indent+1)
    elif isinstance(d, np.ndarray):
        print('\t'*indent, f'{name}, \t, shape={d.shape}, \t sum={d.sum()}')


def remove_nan_inf_cols(arr):
    index = np.isnan(arr).any(axis=0)
    arr = np.delete(arr, index, axis=1)

    index = np.isinf(arr).any(axis=0)
    arr = np.delete(arr, index, axis=1)

    return arr


def get_pls_transform_matrix(pls2):
    mu = pls2._x_mean[np.newaxis, :]
    sigma = pls2._x_std[np.newaxis, :]
    # sklearn版本<1.3 (n_features, n_targets)
    # sklearn版本>=1.3 (n_targets, n_features)
    # w = pls2.coef_
    w = pls2._coef_.T
    b = pls2.intercept_
    assert np.all(sigma == 1)
    Xp = np.vstack([w, b - mu @ w]).T
    return Xp


def num_nonzero_cols(x: np.ndarray):
    sm = np.abs(x).sum(axis=0)
    return np.count_nonzero(sm)


