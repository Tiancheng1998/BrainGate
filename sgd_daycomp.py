import numpy as np
import scipy.io as sio
import os.path as opath
import torch.utils.data as data
import torch
from fastai.basics import *
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from functions import *

# This file compares the performance of sgd logistic classfier across day


is_T9 = True # if true runs T9; else, runs T10
lr=1e-5


if is_T9:
    #T9 / 20161011
    directory1 = opath.join('SLCData_T9', '2016_1011')
    file_suffixes1 = ['124846(8)', '130410(9)', '131937(10)', '133229(11)']
    #T9 / 20160921
    directory2 = opath.join('SLCData_T9', '2016_0921')
    file_suffixes2 = ['131324(7)', '132807(9)', '134530(10)', '135346(11)']
    #T9 / 20160727
    directory3 = opath.join('SLCData_T9', '2016_0727')
    file_suffixes3 = ['131746(8)', '133757(9)', '134726(10)', '141904(11)']
    #T9 / 20160810
    directory4 = opath.join('SLCData_T9', '2016_0810')
    file_suffixes4 = ['124818(7)', '130328(8)', '131712(9)', '132739(10)']
    #T9 / 20161006
    directory5 = opath.join('SLCData_T9', '2016_1006')
    file_suffixes5 = ['125724(7)', '130937(8)', '132435(9)', '134640(11)']
else:
    # T10 0214
    directory1 = opath.join('SLCData_T10', '2017_0214',)
    file_suffixes1 = ['131710(6)', '132443(7)', '133333(8)', '134110(9)']
    # T10 0508
    directory2 = opath.join('SLCData_T10', '2017_0508',)
    file_suffixes2 = ['140023(12)', '140611(13)', '141154(14)', '141654(15)']
    # T10 0509
    directory3 = opath.join('SLCData_T10', '2017_0509',)
    file_suffixes3 = ['133947(11)', '134522(12)', '134957(13)', '135509(14)']
    # T10 0327
    directory4 = opath.join('SLCData_T10', '2017_0327',)
    file_suffixes4 = ['131824(6)', '132856(7)', '135226(11)', '135921(12)']
    # T10 0208
    directory5 = opath.join('SLCData_T10', '2017_0208',)
    file_suffixes5 = ['132036(7)', '132830(8)', '133611(9)', '134612(10)']


directories = [directory2, directory3, directory4, directory5]
file_suffixess = [file_suffixes2, file_suffixes3, file_suffixes4, file_suffixes5]

for d in range(4):
    directory = directories[d]
    file_suffixes = file_suffixess[d]
    if is_T9:
        date = directory[11:]
    else:
        date = directory[12:]

    ncTX = []
    spike_power = []

    for f in file_suffixes:
        p = opath.join(directory, 'SLCdata_' + date + '_' + f)
        mat_data = sio.loadmat(p)
        ncTX.append(mat_data['ncTX']['values'][0, 0].T)
        spike_power.append(mat_data['spikePower']['values'][0, 0].T)

# ncTX and spike_power now should each contain four blocks represented by four matrices (num_channels x num_bins)

# prepare data, for day comparison, we only look at dataset with both rates and power
# concatenate spike rates and spike power along the first dimesion (row)
# concatenate blocks along the second dimension (column)
    full_JACO = np.concatenate([np.concatenate((ncTX[0], spike_power[0]), axis=0),
                                np.concatenate((ncTX[2], spike_power[2]), axis=0)], axis=1)
    full_cursor = np.concatenate([np.concatenate((ncTX[1], spike_power[1]), axis=0),
                                np.concatenate((ncTX[3], spike_power[3]), axis=0)], axis=1)


    # equalizing the amount of training samples for JACO and cursor
    if full_JACO.shape[1] <= full_cursor.shape[1]:
        full_cursor = full_cursor[:, :full_JACO.shape[1]]
    else:
        full_JACO = full_JACO[:, :full_cursor.shape[1]]

    train_JACO = full_JACO
    train_cursor = full_cursor


    results = []
    for win in range(1,51,2):  # loop through all window sizes
        print('window: ', win)
        win_result = []
        # 5 fold cross validation on each window size
        for i_fold in range(5):
            print('fold: ', i_fold)
            train_JACO_1, train_cursor_1, test_JACO, test_cursor = generate_train_test_set(i_fold, train_JACO, train_cursor) # from fuinctions.py file
            train_J = generate_dataset(train_JACO_1, win) # from fuinctions.py file
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
    
    # store the result for each day as its own file
    results_array = np.array([np.array(e) for e in results])
    if is_T9:
        subject = 'T9'
    else:
        subject = 'T10'
    fn = 'win_' + subject + '_logistic_' + date + '.pickle'
    with open(fn, 'wb') as f:
        pickle.dump(results_array, f)

