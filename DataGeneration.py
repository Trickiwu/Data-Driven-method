from pypower import api as pp
from pypower.ext2int import ext2int1

import numpy as np
import numpy.random as nr
from scipy import sparse as ssparse

from pypower.idx_bus import *
from pypower.idx_brch import *
from pypower.idx_gen import *
# from pypower.idx_cost import *
import os

from DLPF import DLPF
import utils




def DataGeneration(case_name, Q_per, data_name, dc_ac, G_range,
                   upper_bound, lower_bound, Q_range, V_range, data_size, L_range,
                   random_load, Va_range, ref, L_corr,
                   verbose=True):
    """
    :param verbose: 控制台是否打印详细信息
    :return:
    """
    # this function generates the data of P, Q, V, and theata
    # define_constants;     # Define matpower constants

    mpc = utils.load_case(case_name)  # export to input
    # utils.recursive_visualize('mpc', mpc)

    num_load = mpc['bus'].shape[0]
    num_branch = mpc['branch'].shape[0]
    num_train = data_size

    # generate the load based on the assumption that the load is random or is correlated
    if random_load == 1:#**
        load_index = nr.random((data_size, num_load)) * (upper_bound - lower_bound) \
                     + lower_bound * np.ones((data_size, num_load))
    elif random_load == 0:
        load_index = nr.random((data_size, 1)) * (upper_bound - lower_bound) \
                     + lower_bound * np.ones((data_size, 1))
        load_index = load_index * np.ones((1, num_load)) + nr.random((data_size, num_load)) * L_range

    X_load = load_index @ np.diag(mpc['bus'][:, PD].T)  # bus matrix;' transpose

    # data generation through power flow calculation, the Matpower Toolbox is required
    data = {
        'P': np.zeros((num_train, num_load)),
       'Va': np.zeros((num_train, num_load))
    }

    if dc_ac:
        data['Q'] = np.zeros((num_train, num_load))
        data['V'] = np.zeros((num_train, num_load))
        data['Va_dc'] = np.zeros((num_train, num_load))
        data['P_dc'] = np.zeros((num_train, num_load))

      
        data['P_dlpf'] = np.zeros((num_train, num_load))
        data['PF_dlpf'] = np.zeros((num_train, mpc['branch'].shape[0]))
        data['Va_dlpf'] = np.zeros((num_train, num_load))
        data['V_dlpf'] = np.zeros((num_train, num_load))
        data['PF_dc'] = np.zeros((num_train, mpc['branch'].shape[0]))

    # mpc['gen'] by default is int64
    mpc['gen'] = mpc['gen'].astype(np.float64)
    gen_ini = mpc['gen'][:, PG].copy()  # must deep copy
    bus_ini = mpc['gen'][:, VG].copy()  # must deep copy

    # Matlab 版本中没有的初始化
    data['PD'] = np.zeros((num_train, mpc['bus'].shape[0]))
    data['QD'] = np.zeros((num_train, mpc['bus'].shape[0]))
    data['PG'] = np.zeros((num_train, mpc['gen'].shape[0]))
    data['QG'] = np.zeros((num_train, mpc['gen'].shape[0]))
    data['PF'] = np.zeros((num_train, mpc['branch'].shape[0]))
    data['PT'] = np.zeros((num_train, mpc['branch'].shape[0]))
    data['QF'] = np.zeros((num_train, mpc['branch'].shape[0]))
    data['QT'] = np.zeros((num_train, mpc['branch'].shape[0]))
    #data['BR_R'] = np.zeros((num_train, mpc['branch'].shape[0]))
    #data['BR_X'] = np.zeros((num_train, mpc['branch'].shape[0]))
    #data['BR_B'] = np.zeros((num_train, mpc['branch'].shape[0]))
    #data['TAP'] = np.zeros((num_train, mpc['branch'].shape[0]))

    for i in range(num_train):
        print(f'------------------------{i}/{num_train}---------------------------')
        mpc['bus'][:, PD] = X_load[i, :].T
        mpc['bus'][:, QD] = mpc['bus'][:, PD] * (Q_per * np.ones(mpc['bus'][:, QD].shape)
                                                 + Q_range * nr.random(mpc['bus'][:, QD].shape))
        mpc['gen'][:, PG] = gen_ini + nr.random(gen_ini.shape) - 0.5 * np.ones(gen_ini.shape) * G_range
        mpc['gen'][:, VG] = bus_ini + (nr.random(bus_ini.shape) - 0.5 * np.ones(bus_ini.shape)) * V_range

        # generate the data based on AC power flow equations or based on DC power flow equations
        if dc_ac:
            # utils.recursive_print_shape('mpc', mpc)
            # matlab: [MVAbase, bus, gen, branch] = runpf(mpc)
            ppopt = pp.ppoption(OUT_ALL=verbose)
            ret_pf, _ = pp.runpf(mpc, ppopt=ppopt)
            MVAbase = ret_pf['baseMVA']
            mpc['bus'][:, VM] = ret_pf['bus'][:, VM]
            I2E, bus, gen, branch = ext2int1(ret_pf['bus'], ret_pf['gen'], ret_pf['branch'])
            # matlab: [MVAbase_dc, bus_dc, gen_dc, branch_dc] = rundcpf(mpc)

            ret_dcpf, _ = pp.rundcpf(mpc, ppopt=ppopt)
            I2E, bus_dc, gen_dc, branch_dc = ext2int1(ret_dcpf['bus'], ret_dcpf['gen'], ret_dcpf['branch'])
            # _, _, data['P_dlpf'][i, :], data['PF_dlpf'][i, :], data['Va_dlpf'][i, :], data['V_dlpf'][i, :] = DLPF(mpc)
            _, _, Pbus, BranchFlow, busAgl, busVol = DLPF(mpc)
            data['P_dlpf'][i, :] = Pbus
            data['PF_dlpf'][i, :] = BranchFlow
            data['Va_dlpf'][i, :] = busAgl
            data['V_dlpf'][i, :] = busVol

            data['Va_dc'][i, :] = ret_dcpf['bus'][:, VA].T

            Sbus_dc = pp.makeSbus(ret_dcpf['baseMVA'], ret_dcpf['bus'], ret_dcpf['gen'])
            data['P_dc'][i, :] = np.real(Sbus_dc).T
            data['PF_dc'][i, :] = branch_dc[:, PF].T
        else:
            # matlab: [MVAbase, bus, gen, branch] = rundcpf(mpc)
            ret_pf, _ = pp.rundcpf(mpc)
            MVAbase, bus = ret_pf['baseMVA'], ret_pf['bus']

        # save the generation results into the data struct
        Sbus = pp.makeSbus(ret_pf['baseMVA'], ret_pf['bus'], ret_pf['gen'])
        data['P'][i, :] = np.real(Sbus).T
        data['Q'][i, :] = np.imag(Sbus).T
        data['V'][i, :] = bus[:, VM].T
        data['Va'][i, :] = bus[:, VA].T
        data['PG'][i, :] = gen[:, PG]
        data['QG'][i, :] = gen[:, QG]
        data['PD'][i, :] = bus[:, PD]
        data['QD'][i, :] = bus[:, QD]
        data['PF'][i, :] = branch[:, PF].T
        data['PT'][i, :] = branch[:, PT].T
        data['QF'][i, :] = branch[:, QF].T
        data['QT'][i, :] = branch[:, QT].T
        #data['BR_R'][i, :] = branch[:, BR_R]
        #data['BR_X'][i, :] = branch[:, BR_X]
        #data['BR_B'][i, :] = branch[:, BR_B]
        #data['TAP'][i, :] = branch[:, TAP]


    mpc_int = pp.ext2int(mpc)
    B = pp.makeBdc(MVAbase, bus, mpc_int['branch'])[0]
    if ssparse.issparse(B):
        B = B.toarray()
    # save to files
    # eval(['save ', data_name, ' data num_load num_branch B;'])
    if not os.path.exists('data'):
        os.mkdir('data')
    np.savez_compressed(os.path.join('data', data_name),
                        data=data,
                        num_load=num_load,
                        num_branch=num_branch,
                        B=B
                        )
