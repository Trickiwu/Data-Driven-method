import pypower.api as pp

import numpy as np
from numpy import ix_
import numpy.random as nr
import scipy.sparse as ssparse
from pypower.bustypes import bustypes
from pypower.idx_bus import *
from pypower.idx_brch import *
from pypower.idx_gen import *
from pypower.idx_cost import *
from pypower.makeSbus import makeSbus
from pypower.makeYbus import makeYbus
import utils

from typing import Tuple


def DLPF(mpc) -> Tuple[ssparse.spmatrix, ssparse.spmatrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # This code realizes Decoupled Linearized Power Flow (DLPF),this code is from the publication of "Yang J, Zhang N, Kang C, et al. A State-Independent Linear Power Flow Model with Accurate Estimation of Voltage Magnitude[J]. IEEE Transactions on Power Systems, 2016, PP(99):1-1."

    # Define matpower constants
    # define_constants
    # Load case
    # pypower
    mpc = pp.ext2int(mpc)
    # utils.recursive_visualize('mpc', mpc)
    baseMVA, bus, gen, branch = mpc['baseMVA'], mpc['bus'], mpc['gen'], mpc['branch']
    ref, pv, pq = bustypes(bus, gen)

    # Matrices
    Ybus, _, _ = makeYbus(baseMVA, bus, branch)  ## matpower case
    Gbus = np.real(Ybus)
    Bbus = np.imag(Ybus)
    GP = Gbus
    BD = np.diag(Bbus.sum(0))  # shunt elements
    BP = -Bbus + BD
    BQ = -Bbus
    GQ = -Gbus  # GQ approxiately equals -Gbus
    Sbus = makeSbus(baseMVA, bus, gen)
    Pbus = np.real(Sbus)
    Qbus = np.imag(Sbus)

    #
    Xp_dlpf = ssparse.hstack([BP, GP])
    Xq_dlpf = ssparse.hstack([GQ, BQ])

    # Matrices in equation (11)
    #      | Pm1 |   | B11  G12 |   | delta   |
    #      |     | = |          | * |         |
    #      | Qm1 |   | G21  B22 |   | voltage |
    pv_pq = np.concatenate([pv, pq])
    B11 = BP[ix_(pv_pq, pv_pq)]
    G12 = GP[ix_(pv_pq, pq)]
    G21 = GQ[ix_(pq, pv_pq)]
    B22 = BQ[ix_(pq, pq)]
    Pm1 = Pbus[pv_pq] - GP[ix_(pv_pq, ref)] * bus[ref, VM] - GP[ix_(pv_pq, pv)] * bus[pv, VM]
    Qm1 = Qbus[pq] - BQ[ix_(pq, ref)] * bus[ref, VM] - BQ[ix_(pq, pv)] * bus[pv, VM]

    # Matrices in equation (17)
    t1 = utils.right_solve(G12, B22)
    t2 = utils.right_solve(G21, B11)
    B11m = B11 - t1 @ G21
    B22m = B22 - t2 @ G12
    Pm2 = Pm1 - t1 @ Qm1
    Qm2 = Qm1 - t2 @ Pm1

    # Calculate voltage magnitude
    n = bus.shape[0]
    busVol = np.zeros((n,))    
    busVol[ref] = bus[ref, VM]
    busVol[pv] = bus[pv, VM]
    busVol[pq] = utils.left_solve(B22m, Qm2)
    # Calculate voltage phase angle
    busAgl = np.ones((n,)) * bus[ref, VA]
    busAgl[pv_pq] = busAgl[pv_pq] + utils.left_solve(B11m, Pm2) / np.pi * 180
    # Calculate line lossless MW flow
    branch[:, TAP] = 1

    # BranchFlow1 = (busVol(branch(:,F_BUS))./branch(:,TAP)-busVol(branch(:,T_BUS)))
    #               .*branch(:,BR_R)
    #               ./(branch(:,BR_X).^2+branch(:,BR_R).^2)*baseMVA
    BranchFlow1_a = busVol[branch[:, F_BUS].astype(np.uint32)] / branch[:, TAP] - busVol[branch[:, T_BUS].astype(np.uint32)]
    BranchFlow1_b = branch[:, BR_R]
    BranchFlow1_c = branch[:, BR_X] ** 2 + branch[:, BR_R] ** 2
    BranchFlow1 = BranchFlow1_a * BranchFlow1_b / BranchFlow1_c * baseMVA

    # BranchFlow2 = (busAgl(branch(:,F_BUS))-busAgl(branch(:,T_BUS)))
    #               .*branch(:,BR_X)
    #               ./(branch(:,BR_X).^2+branch(:,BR_R).^2)./branch(:,TAP)/180*pi*baseMVA
    BranchFlow2_a = busAgl[branch[:, F_BUS].astype(np.uint32)] - busAgl[branch[:, T_BUS].astype(np.uint32)]
    BranchFlow2_b = branch[:, BR_X]
    BranchFlow2_c = branch[:, BR_X] ** 2 + branch[:, BR_R] ** 2
    BranchFlow2_d = branch[:, TAP]
    BranchFlow2 = BranchFlow2_a * BranchFlow2_b / BranchFlow2_c / BranchFlow2_d / 180 * np.pi * baseMVA

    BranchFlow = BranchFlow1 + BranchFlow2

    return Xp_dlpf, Xq_dlpf, Pbus, BranchFlow, busAgl, busVol
