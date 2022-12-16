import numpy as np
from numpy import ix_

import utils


def assert_1d_vector(arr):
    assert arr.ndim == 1, f'不满足1d列向量，输入数据尺寸是shape={arr.shape}' \
                          f'索引(index)向量必须是1d形式，即(n, ) '


def assert_1d_vectors(arrs):
    for arr in arrs:
        assert_1d_vector(arr)


def assert_2d_col_vector(arr):
    assert arr.ndim == 2 and arr.shape[1] == 1, f'不满足2d列向量，输入数据尺寸是shape={arr.shape}, ' \
                                          f'数据(data)向量全部是2d形式，即(n, 1) '


def assert_2d_col_vectors(arrs):
    for arr in arrs:
        assert_2d_col_vector(arr)


def TestAccuracyInverse(num_train, data, Xv, Xva, ref, pv, pq, num_load, extras):
    # this function test the accuracy of inverse regression
    # note that the regression matrix is reordered
    #   |Va_pq |    |         ||P_pq |    |  |
    #   |Va_pv |    |X11  X12 ||P_pv |    |C1|
    #   |P_ref |    |         ||1    |    |  |
    #   |V_pq  | =  |         ||Q_pq | +  |  |
    #   |      |    |         ||     |    |  |
    #   |V_pv  |    |X21  X22 ||Q_pv |    |C2|
    #   |V_ref |    |         ||Q_ref|    |  |
    #   Y = X * a
    # pq, pv, ref are scalars, so using np.concatenate
    pq_pv = np.concatenate([pq, pv])
    pv_ref = np.concatenate([pv, ref])
    pq_pv_ref = np.concatenate([pq, pv, ref])
    pq_pv_ref_pqnumload = np.concatenate([pq, pv, ref, pq + num_load])
    assert_1d_vector(pq_pv)
    assert_1d_vector(pv_ref)
    assert_1d_vector(pq_pv_ref)
    assert_1d_vector(pq_pv_ref_pqnumload)

    X11 = np.vstack([Xva[ix_(pq_pv_ref, pq_pv_ref_pqnumload)],
                     Xv[ix_(pq, pq_pv_ref_pqnumload)]])

    X12 = np.vstack([Xva[ix_(pq_pv_ref, pv_ref + num_load)],
                     Xv[ix_(pq, pv_ref + num_load)]])

    X21 = Xv[ix_(pv_ref, pq_pv_ref_pqnumload)]
    X22 = Xv[ix_(pv_ref, pv_ref + num_load)]


    idx_2numload = [2 * num_load]
    C1 = np.vstack([Xva[ix_(pq_pv_ref, idx_2numload)],
                    Xv[ix_(pq, idx_2numload)]])
    C2 = Xv[ix_(pv_ref, idx_2numload)]
    assert_2d_col_vector(C1)
    assert_2d_col_vector(C2)

    P = data['P'].copy()
    P[:, ref] = data['Va'][:, ref].copy()
    Va = data['Va'].copy()
    Va[:, ref] = data['P'][:, ref].copy()

    #
    data['V_fitting'] = np.zeros((num_train, num_load))
    data['Va_fitting'] = np.zeros((num_train, num_load))
    data['P_fitting'] = np.zeros((num_train, num_load))
    data['Q_fitting'] = np.zeros((num_train, num_load))

    delta = {
        'va': {
            'fitting': None,
            'dlpf': None
        },
        'v': {
            'fitting': None,
            'dlpf': None
        },
        'pf': {
            'dcpf': None,
            'dlpf': None
        }
    }

    # # calculate the results by data - driven linearized equations
    for i in range(num_train):

        Y2 = data['V'][i, pv_ref][:, np.newaxis].copy()    # [n, 1]
        assert_2d_col_vector(Y2)

        a1 = np.concatenate([P[i, pq_pv_ref], data['Q'][i, pq].T])[:, np.newaxis]  # # x1
        a2 = utils.left_solve(X22, Y2 - X21 @ a1 - C2)  # # x2
        assert_2d_col_vector(a2)

        num_pq = pq.shape[0]  # # 1
        num_pv = pv.shape[0]  # # 3
        Q_pv = a2[0:num_pv]  # # 1: 3
        Q_ref = a2[num_pv:num_pv + 1]  # # 4: 4

        Y1 = X11 @ a1 + X12 @ a2 + C1   # 2d
        assert_2d_col_vector(Y1)

        # 1d向量
        V = np.zeros((num_load, 1))
        Va = np.zeros((num_load, 1))
        Q = data['Q'][i][:, np.newaxis].copy()
        assert_2d_col_vector(Y1)
        assert_2d_col_vector(Va)
        assert_2d_col_vector(Q)

        V[pv_ref] = data['V'][i, pv_ref][:, np.newaxis].copy()  # # V of pv ref bus
        V[pq] = Y1[num_load: num_load + num_pq].copy()  # # Y1 = [Ang(L), Ang(S), P(r), V(L)]T 6: 6
        Va[ref] = data['Va'][i, ref][:, np.newaxis].copy()
        Va[pq_pv] = Y1[0: num_pq + num_pv].copy() / np.pi * 180
        P[i, ref] = Y1[num_pq + num_pv + 1].copy()
        Q[pv_ref] = np.vstack([Q_pv, Q_ref])

        data['V_fitting'][i, :] = V.T
        data['Va_fitting'][i, :] = Va.T
        data['P_fitting'][i, :] = P[i, :]
        data['Q_fitting'][i, :] = Q.T

    # # # calculate the errors, note that the value of nan or inf is removed
    temp = np.abs(data['Va'] - data['Va_fitting'])
    temp = utils.remove_nan_inf_cols(temp)
    delta['va']['fitting'] = temp.mean()

    temp = np.abs(data['V'][:, pq] - data['V_fitting'][:, pq])
    temp = utils.remove_nan_inf_cols(temp)
    delta['v']['fitting'] = temp.mean()

    temp = np.abs(data['Va'] - data['Va_dlpf'])
    temp = utils.remove_nan_inf_cols(temp)
    delta['va']['dlpf'] = temp.mean()

    temp = np.abs(data['V'][:, pq] - data['V_dlpf'][:, pq])
    temp = utils.remove_nan_inf_cols(temp)
    delta['v']['dlpf'] = temp.mean()

    temp = np.abs((data['PF'] - data['PF_dc']) / data['PF'])
    temp = utils.remove_nan_inf_cols(temp)
    delta['pf']['dcpf'] = temp.mean() * 100

    temp = np.abs((data['PF'] - data['PF_dlpf']) / data['PF'])
    temp = utils.remove_nan_inf_cols(temp)
    delta['pf']['dlpf'] = temp.mean() * 100

    return data, delta