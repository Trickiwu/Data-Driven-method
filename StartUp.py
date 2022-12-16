
import os
import numpy as np

import pypower.api as pp

from DataGeneration import DataGeneration
from DLPF import DLPF
from RegressionForward import RegressionForward
from RegressionInverse import RegressionInverse
import utils
from TestAccuracyForward import TestAccuracyForward
from TestAccuracyInverse import TestAccuracyInverse

generate_data = 1 # 1,data generation is needed; 0,data is already generated
generate_test_data = 1  # 1,data generation is needed; 0,data is already generated
upper_bound = 1.2  # upper bound of generated load
lower_bound = 0.8  # lower bound of generated load
regression = 1  # 0-least squares 1-pls regression 2-bayesian linear regression
for_or_inv = 0 # 0-forward regression;1-inverse regression

G_range = 0.1  # range of power generation variations
Q_range = 0.25  # range of Q variations
Q_per = 0.2  # Q percentage on P
V_range = 0.01  # range of voltage magnitude variations of PV buses
L_range = 0.05  # range of load in different nodes
L_corr = 0.9  # covariance
Va_range = 7  # degree
Va_num = []
dc_ac = 1  # 0-dc;1-ac;
random_load = 1  # 1,random 0,not random with bounder 2,not random with covariance

data_size = 500  # training data size
data_size_test = 300  # testing data size
case_name = 'case30'
address = ''  # address to read and save the data filess

# training data generation
data_name = os.path.join(address, case_name + '_training_data')
print('data_name:', data_name)
if generate_data:
    mpc = utils.load_case(case_name)
    mpc = pp.ext2int(mpc)
    ref, pv, pq = pp.bustypes(mpc['bus'], mpc['gen'])
    DataGeneration(case_name, Q_per, data_name, dc_ac, G_range,
                   upper_bound, lower_bound, Q_range, V_range, data_size, L_range,
                   random_load, Va_range, ref, L_corr,
                   verbose=True)

with np.load(f'data/{data_name}.npz', allow_pickle=True) as npz:
    data = npz['data'].item()
    num_load = npz['num_load']
    num_branch = npz['num_branch']
    B = npz['B']

    print(f'loaded data are:'
          f'num_branch:{num_branch}, '
          f'num_load:{num_load}, '
          f'data:{data.keys()}, '
          f'B:{B.shape}')

# linear regression
# get bus index lists of each type of bus
mpc = utils.load_case(case_name)
mpc = pp.ext2int(mpc)
ref, pv, pq = pp.bustypes(mpc['bus'], mpc['gen'])

Xp_dlpf, Xq_dlpf, _, _, _, _ = DLPF(mpc)  # ignore  3.4.5 parameters
Xp_dlpf = Xp_dlpf.todense()
Xq_dlpf = Xq_dlpf.todense()

if for_or_inv == 0:
    Xp, Xq, Xpf, Xqf, Xpt, Xqt, X_pls, k, extras = RegressionForward(regression, num_load, data, address, case_name)
else:
    Xva, Xv, Xpf, Xqf, extras = RegressionInverse(regression, num_load, data, ref, address, case_name)

# generate testing data
upper_bound = 1.2
lower_bound = 0.8
data_name = os.path.join(address, case_name + '_testing_data')
print('testing_data_name:', data_name)
if generate_test_data:
    DataGeneration(case_name, Q_per, data_name, dc_ac, G_range,
                   upper_bound, lower_bound, Q_range, V_range, data_size_test, L_range,
                   random_load, Va_range, ref, L_corr,
                   verbose=True)

with np.load(f'data/{data_name}.npz', allow_pickle=True) as npz:
    data = npz['data'].item()
    num_load = npz['num_load']
    num_branch = npz['num_branch']
    B = npz['B']

    print(f'loaded data are:'
          f'num_branch:{num_branch}, '
          f'num_load:{num_load}, '
          f'data:{data.keys()}, '
          f'B:{B.shape}')

    num_train = data['P'].shape[0]

#  verify the accuracy
if for_or_inv == 0:
    delta, test = TestAccuracyForward(num_train, data, Xp, Xq,Xpf,Xqf, Xp_dlpf, Xq_dlpf, B, extras)
else:
    ref, pv, pq = pp.bustypes(mpc['bus'], mpc['gen'])
    data, delta = TestAccuracyInverse(num_train, data, Xv, Xva, ref, pv, pq, num_load, extras)

print(delta)
