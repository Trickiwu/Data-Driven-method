import numpy as np

import utils


def TestAccuracyForward(num_train, data, Xp, Xq,Xpf,Xqf, Xp_dlpf, Xq_dlpf, B, extras):
    """
    this function test the accuracy of forward regression
    calculate the results by data-driven linearized equations
    :param num_train:
    :param data:
    :param Xp:
    :param Xq:
    :param Xp_dlpf:
    :param Xq_dlpf:
    :param B:
    :return:
    """

    cols = Xp.T.shape[1]
    cols1 = Xpf.T.shape[1]
    test = {
        'p': {
            'fitting': np.zeros((num_train, cols)),
            'dcpf': np.zeros((num_train, cols)),
            'dlpf': np.zeros((num_train, cols))
        },
        'q': {
            'fitting': np.zeros((num_train, cols)),
            'dlpf': np.zeros((num_train, cols))
        },
        'pf': {
            'fitting': np.zeros((num_train, cols1))
        },
        'qf': {
            'fitting': np.zeros((num_train, cols1))
        }
    }
    delta = {
        'p': {
            'fitting': None,
            'dcpf': None,
            'dlpf': None
        },
        'q': {
            'fitting': None,
            'dlpf': None
        },
        'pf': {
            'fitting': None
        },
        'qf': {
            'fitting': None
        }
    }
    for i in range(num_train):
        Va = data['Va'][i, :].copy()
        V_Va = np.hstack([data['Va'][i, :] * np.pi / 180, data['V'][i, :]])
        V_Va_1 = np.hstack([V_Va, 1])

        test['p']['fitting'][i, :] = V_Va_1 @ Xp.T
        test['p']['dcpf'][i, :] = B @ Va * np.pi / 180
        test['p']['dlpf'][i, :] = V_Va @ Xp_dlpf.T
        test['pf']['fitting'][i, :] = V_Va_1 @ Xpf.T

        test['q']['fitting'][i, :] = V_Va_1 @ Xq.T
        test['q']['dlpf'][i, :] = V_Va @ Xq_dlpf.T
        test['qf']['fitting'][i, :] = V_Va_1 @ Xqf.T

    # calculate the errors, note that the value of nan or inf is removed
    temp = np.absolute((data['P'] - test['p']['fitting']) / data['P'])
    temp = utils.remove_nan_inf_cols(temp)
    delta['p']['fitting'] = temp.mean() * 100

    temp = np.absolute((data['P'] - test['p']['dcpf']) / data['P'])
    temp = utils.remove_nan_inf_cols(temp)
    delta['p']['dcpf'] = temp.mean() * 100

    temp = np.absolute((data['P'] - test['p']['dlpf']) / data['P'])
    temp = utils.remove_nan_inf_cols(temp)
    delta['p']['dlpf'] = temp.mean() * 100

    temp = np.absolute((data['Q'] - test['q']['fitting']) / data['Q'])
    temp = utils.remove_nan_inf_cols(temp)
    delta['q']['fitting'] = temp.mean() * 100

    temp = np.absolute((data['Q'] - test['q']['dlpf']) / data['Q'])
    temp = utils.remove_nan_inf_cols(temp)
    delta['q']['dlpf'] = temp.mean() * 100

    temp = np.absolute((data['PF'] - test['pf']['fitting']) / data['PF'])
    temp = utils.remove_nan_inf_cols(temp)
    delta['pf']['fitting'] = temp.mean() * 100

    temp = np.absolute((data['QF'] - test['qf']['fitting']) / data['QF'])
    temp = utils.remove_nan_inf_cols(temp)
    delta['qf']['fitting'] = temp.mean() * 100
    return delta, test
