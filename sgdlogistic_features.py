import numpy as np
import scipy.io as sio
import os.path as opath
import torch.utils.data as data
import torch
from sklearn.linear_model import SGDClassifier
from ContinuousDataset import ContinuousDataset
import matplotlib.pyplot as plt
from functions import *

# This file compares the performance of logistic classifier on different neural features (rates and power)
# this script produces a pickle file containing the accuracy matrix. change the boolean below to change the neural feature data.
rates_and_power = True
rates_only = False
power_only = False


is_T9 = True # if true runs T9; else, runs T10
# specifies which feature to run on, choose one.
lr = 1e-5


## Load file ##
if is_T9:
    directory = opath.join('SLCData_T9', '2016_1011',)
    file_suffixes = ['124846(8)', '130410(9)', '131937(10)', '133229(11)'] # each file is a block
    date = directory[11:]
else:
    directory = opath.join('SLCData_T10', '2017_0214',)
    file_suffixes = ['131710(6)', '132443(7)', '133333(8)', '134110(9)']
    date = directory[12:]



ncTX = []
spike_power = []


for f in file_suffixes1:
    p = opath.join(directory1, 'SLCdata_' + date + '_' + f)
    mat_data = sio.loadmat(p)
    ncTX.append(mat_data['ncTX']['values'][0, 0].T)
    spike_power.append(mat_data['spikePower']['values'][0, 0].T)


# prepare data, extract neural features that will run on
# concatenate spike rates and spike power along the first dimesion (row)
# concatenate blocks along the second dimension (column)
if rates_and_power:
    full_JACO = np.concatenate([np.concatenate((ncTX[0], spike_power[0]), axis=0),
                                np.concatenate((ncTX[2], spike_power[2]), axis=0)], axis=1)

    full_cursor = np.concatenate([np.concatenate((ncTX[1], spike_power[1]), axis=0),
                                np.concatenate((ncTX[3], spike_power[3]), axis=0)], axis=1)
else:
    if rates_only:
        full_JACO = np.concatenate([ncTX[0], ncTX[2]], axis=1).astype('float64')
        full_cursor = np.concatenate([ncTX[1], ncTX[3]], axis=1).astype('float64')
    else:
        full_JACO = np.concatenate([spike_power[0], spike_power[2]], axis=1).astype('float64')
        full_cursor = np.concatenate([spike_power[1], spike_power[3]], axis=1).astype('float64')


# equalizing the amount of training samples for JACO and cursor
if full_JACO.shape[1] <= full_cursor.shape[1]:
    full_cursor = full_cursor[:, :full_JACO.shape[1]]
else:
    full_JACO = full_JACO[:, :full_cursor.shape[1]]

train_JACO = full_JACO
train_cursor = full_cursor

# loop through different window lengths
results = []
for win in range(1,51,2):
    print('window: ', win)
    win_result = []
    for i_fold in range(5):
        print('fold: ', i_fold)
        train_JACO_1, train_cursor_1, test_JACO, test_cursor = generate_train_test_set(i_fold, train_JACO, train_cursor) # find this function in the function file
        train_J = generate_dataset(train_JACO_1, win)
        train_c = generate_dataset(train_cursor_1, win)
        train_X = np.concatenate([train_J, train_c])
        train_Y = np.concatenate([np.zeros(train_J.shape[0]), np.ones(train_c.shape[0])])
        test_J = generate_dataset(test_JACO, win)
        test_c = generate_dataset(test_cursor, win)
        test_X = np.concatenate([test_J, test_c])
        test_Y = np.concatenate([np.zeros(test_J.shape[0]), np.ones(test_c.shape[0])])

        model = SGDClassifier(penalty='none', learning_rate='adaptive', loss='log', eta0=lr)
        model.fit(train_X, train_Y)
        preds = model.predict(test_X)
        print('accuracy: ', np.sum(preds == test_Y) / len(test_Y))
        win_result.append(np.sum(preds == test_Y) / len(test_Y))

    results.append(win_result)

results_array = np.array([np.array(e) for e in results])
fn = 'win_T10_logistic_power.pickle'
with open(fn, 'wb') as f:
    pickle.dump(results_array, f)
# The file saved should contain a matrix of dimension (20 x 5) 20 is window length. 5 is 5 fold.


# Below plots the results. text is inaccuracte
plt.plot(np.array(range(1,51,2)) * 20, results_array.mean(axis=1), color='b',label='power only')
plt.xlabel('window length (ms)')
plt.ylabel('accuracy')
plt.legend()
plt.title('logistic classifier accuracy vs window length T9')
plt.show()
