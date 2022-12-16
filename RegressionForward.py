import numpy as np
import numpy.random as nr
from sklearn.cross_decomposition import PLSRegression
from bayesianLR import bayesian_lr
import utils


def RegressionForward(regression, num_load, data, address, case_name):
    """
    this function conduct the forward regression by calling different regression algorithms
    switch regression
    :param regression:
    :param num_load:
    :param data:
    :param address:
    :param case_name:
    :return:
    """
    cols = data['Va'].shape[1] + data['V'].shape[1] + 1
    Xp = np.zeros((num_load, cols))
    Xq = np.zeros((num_load, cols))
    Xqf = np.zeros((num_load, cols))
    Xpf = np.zeros((num_load, cols))

    X_pls = None
    k = None
    if regression == 0:  # ordinary least squares
        for i in range(num_load):
            p = data['P'][:, i]
            V_Va_p = np.hstack([data['Va'] * np.pi / 180, data['V'], np.ones((data['V'].shape[0], 1))])
            b, _, _, _ = np.linalg.lstsq(V_Va_p, p)  # b
            Xp[i, :] = b.T

            q = data['Q'][:, i]
            V_Va_q = np.hstack([data['Va'] * np.pi / 180, data['V'], np.ones((data['V'].shape[0], 1))])
            b, _, _, _ = np.linalg.lstsq(V_Va_q, q)
            Xq[i, :] = b.T

        Xpf = []
        Xqf = []
        Xpt = []
        Xqt = []
        extras = None

    elif regression == 1:  # partial least squares
        k = np.linalg.matrix_rank(data['V']) + np.linalg.matrix_rank(data['Va'])
        k = min(k, data['P'].shape[0] - 1)  # former: m latter: n - 1ncomp
        X_pls = np.hstack([data['Va'] * np.pi / 180, data['V']])  # X = n * m

        Y_p_pls = data['P']  # 300 * 5 n * p
        pls_regressor_Xp = PLSRegression(n_components=k, scale=False)
        pls_regressor_Xp.fit(X_pls, Y_p_pls)    # k=9 p+1*m  10*5
        Xp = utils.get_pls_transform_matrix(pls_regressor_Xp)

        Y_q_pls = data['Q']
        pls_regressor_Xq = PLSRegression(n_components=k, scale=False)
        pls_regressor_Xq.fit(X_pls, Y_q_pls)
        Xq = utils.get_pls_transform_matrix(pls_regressor_Xq)

        k = min(k, data['PF'].shape[0] - 1)  # former: m latter: n - 1ncomp
        X_pls = np.hstack([data['Va'] * np.pi / 180, data['V']])  # X = n * m

        Y_pf_pls = data['PF']  # 300 * 5 n * p
        pls_regressor_Xpf = PLSRegression(n_components=k, scale=False)
        pls_regressor_Xpf.fit(X_pls, Y_pf_pls)  # k=9 p+1*m  10*5
        Xpf = utils.get_pls_transform_matrix(pls_regressor_Xpf)

        Y_qf_pls = data['QF']
        pls_regressor_Xqf = PLSRegression(n_components=k, scale=False)
        pls_regressor_Xqf.fit(X_pls, Y_qf_pls)
        Xqf = utils.get_pls_transform_matrix(pls_regressor_Xqf)

        Xpt = []
        Xqt = []
        extras = None
    elif regression == 2:  # bayesian linear regression
        threshold = 10000
        X = np.hstack([data['Va'] * np.pi / 180, data['V']])
        Y = np.hstack([data['P'], data['Q']])

        X_blr = bayesian_lr(X, Y, threshold)

        row = X_blr.shape[0] // 2
        Xp = X_blr[:row, :]
        Xq = X_blr[row:2*row, :]

        X = np.hstack([data['Va'] * np.pi / 180, data['V']])
        Y1 = np.hstack([data['PF'], data['QF']])

        X1_blr = bayesian_lr(X, Y1, threshold)

        row = X1_blr.shape[0] // 2
        Xpf = X1_blr[:row, :]
        Xqf = X1_blr[row:2 * row, :]

        Xpt = []
        Xqt = []
        extras = None
    else:
        print('no such regression method')

    return Xp, Xq, Xpf, Xqf, Xpt, Xqt, X_pls, k, extras
