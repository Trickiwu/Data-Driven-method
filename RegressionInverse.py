import numpy as np
from sklearn.cross_decomposition import PLSRegression
from bayesianLR import bayesian_lr

import utils


def RegressionInverse(regression, num_load, data, ref, address, case_name):
    """
    this function conduct the inverse regression by calling different regression algorithms
    :param regression:
    :param num_load:
    :param data:
    :param ref:
    :param address:
    :param case_name:
    :param is_sigma:
    :return:
    """
    cols = data['P'].shape[1] + data['Q'].shape[1] + 1
    Xva = np.zeros((num_load, cols))
    Xv = np.zeros((num_load, cols))

    if regression == 0:  # % ordinary least squares
        for i in range(num_load):
            P = data['P'].copy()
            P[:, ref] = 0
            PQ_va = np.hstack([P, data['Q'], np.ones((data['V'].shape[0], 1))])
            b, _, _, _ = np.linalg.lstsq(PQ_va, data['Va'][:, i] * np.pi / 180)
            Xva[i, :] = b.T

            b, _, _, _ = np.linalg.lstsq(PQ_va, data['V'][:, i])
            Xv[i, :] = b.T
        Xpf = []
        Xqf = []
        extras = None
    elif regression == 1:
        # -------------deepcopy
        P = data['P'].copy()
        P[:, ref] = 0
        k = np.linalg.matrix_rank(P) + np.linalg.matrix_rank(data['Q']) + 1
        k = min(k, data['P'].shape[0] - 1)
        X_pls = np.hstack([P, data['Q']])  # 500 * 10 n * p
        k = min(k, utils.num_nonzero_cols(X_pls))

        # Va
        Y_va_pls = data['Va'] * np.pi / 180  # 500 * 5n * m
        Y_va_pls[:, ref] = data['P'][:, ref].copy()
        pls_regressor_Xva = PLSRegression(n_components=k, scale=False)
        pls_regressor_Xva.fit(X_pls, Y_va_pls)
        Xva = utils.get_pls_transform_matrix(pls_regressor_Xva)

        # V
        Y_v_pls = data['V']
        pls_regressor_Xv = PLSRegression(n_components=k, scale=False)
        pls_regressor_Xv.fit(X_pls, Y_v_pls)
        Xv = utils.get_pls_transform_matrix(pls_regressor_Xv)

        Y_pf_pls = data['PF']
        pls_regressor_Xpf = PLSRegression(n_components=k, scale=False)
        pls_regressor_Xpf.fit(X_pls, Y_pf_pls)
        Xpf = utils.get_pls_transform_matrix(pls_regressor_Xpf)

        Y_qf_pls = data['QF']
        pls_regressor_Xqf = PLSRegression(n_components=k, scale=False)
        pls_regressor_Xqf.fit(X_pls, Y_qf_pls)
        Xqf = utils.get_pls_transform_matrix(pls_regressor_Xqf)

        extras = None
    elif regression == 2:  # bayesian linear regression
        threshold = 900000
        P = data['P'].copy()
        P[:, ref] = 0
        X = np.hstack([P, data['Q']])
        Y_va_pls = data['Va'] * np.pi / 180
        Y_va_pls[:, ref] = data['P'][:, ref].copy()
        Y_v_pls = data['V']
        Y = np.hstack([Y_va_pls, Y_v_pls])
        X_blr = bayesian_lr(X, Y, threshold)

        row = X_blr.shape[0] // 2
        Xva = X_blr[0:row, :]
        Xv = X_blr[row:2 * row, :]

        Y_pf_pls = data['PF']
        Y_qf_pls = data['QF']
        Y = np.hstack([Y_pf_pls, Y_qf_pls])
        X_blr = bayesian_lr(X, Y, threshold)

        row = X_blr.shape[0] // 2
        Xpf = X_blr[0:row, :]
        Xqf = X_blr[row:2 * row, :]
        extras = None
    else:
        print('no such regression method')

    return Xva, Xv, Xpf, Xqf, extras
