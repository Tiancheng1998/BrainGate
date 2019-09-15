import numpy as np
import scipy.io as sio
import os.path as opath
import torch.utils.data as data
import torch
from fastai.basics import *
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from ContinuousDataset import ContinuousDataset
import matplotlib.pyplot as plt


is_T9 = True # if true runs T9; else, runs T10
win = 20
lr = 1e-5


# This script investigates the logistic classifier's robustness against channel drop (info loss)

# we do not need five fold cross validation, do not need to divide the dataset into five folds
def generate_train_test_set(JACO, cursor): 
    assert(JACO.shape[1] == cursor.shape[1])
    block_size = JACO.shape[1]
    test_start = math.floor(block_size*0.8)
    test_JACO = JACO[:, test_start:block_size]
    train_JACO = np.delete(JACO, np.arange(test_start, block_size), axis=1)
    test_cursor = cursor[:, test_start:block_size]
    train_cursor = np.delete(cursor, np.arange(test_start, block_size), axis=1)

    return train_JACO, train_cursor, test_JACO, test_cursor


def generate_dataset(ds_m, window):
    ds = []
    i = 0
    while i + window < ds_m.shape[1]:
        ds.append(ds_m[:, i:i+window].flatten())
        i += 1
    return np.stack(ds)


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


for f in file_suffixes:
    p = opath.join(directory, 'SLCdata_' + date + '_' + f)
    mat_data = sio.loadmat(p)
    ncTX.append(mat_data['ncTX']['values'][0, 0].T)
    spike_power.append(mat_data['spikePower']['values'][0, 0].T)


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

# the code below loops through the number of channels dropped
# at each loop, it randomly drops the specified number of channels, train/test the classifier and repeat it five times
results = []
for channel_dropped in range(0,190,5):
    print('channel dropped: ', channel_dropped)
    ch_result = []

    for i in range(5):
        print('fold: ', i)
        channel_drop_list = np.random.permutation(192)[:channel_dropped] # generate a list of channel indices which will be dropped
        new_channels = np.concatenate([np.delete(np.arange(192), channel_drop_list),
                                      np.delete(np.arange(192), channel_drop_list) + 192]) # generate a list of channel indices excluding the dropped channel, do so for both rates and power. 
        train_JACO_select = train_JACO[new_channels] # apply the list of chhannel to get neural data after channel drop
        train_cursor_select = train_cursor[new_channels]
        train_JACO_1, train_cursor_1, test_JACO, test_cursor = generate_train_test_set(train_JACO_select, train_cursor_select) # train test split
        train_J = generate_dataset(train_JACO_1, win) # slide the window and get samples for classifier
        train_c = generate_dataset(train_cursor_1, win)
        train_X = np.concatenate([train_J, train_c]) # concatenate jaco with cursor to get the entire dataset
        train_Y = np.concatenate([np.zeros(train_J.shape[0]), np.ones(train_c.shape[0])]) # generate label
        test_J = generate_dataset(test_JACO, win)
        test_c = generate_dataset(test_cursor, win)
        test_X = np.concatenate([test_J, test_c])
        test_Y = np.concatenate([np.zeros(test_J.shape[0]), np.ones(test_c.shape[0])])

        model = SGDClassifier(penalty='none', learning_rate='adaptive', loss='log', eta0=lr)
        model.fit(train_X, train_Y)
        preds = model.predict(test_X)
        print('accuracy: ', np.sum(preds == test_Y) / len(test_Y))
        ch_result.append(np.sum(preds == test_Y) / len(test_Y))

    results.append(ch_result)

# store the results: accuracy matrix (39 x 5) 
results_array = np.array([np.array(e) for e in results])
if is_T9:
        subject = 'T9'
    else:
        subject = 'T10'
fn = subject + '_logistic_chdrop.pickle'
with open(fn, 'wb') as f:
    pickle.dump(results_array, f)

# below plots the results, text inaccuracte
plt.errorbar(range(0,190,5), results_array.mean(axis=1), yerr= results_array.std(axis=1), color='b',ecolor='g', label='T10')
plt.xlabel('number of channels dropped')
plt.ylabel('accuracy')
plt.title('logistic classifier accuracy vs number of channels dropped')
plt.legend()
plt.show()