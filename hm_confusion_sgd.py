import numpy as np
import scipy.io as sio
import os.path as opath
import torch.utils.data as data
import torch
from ContinuousDataset import ContinuousDataset
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.linear_model import LogisticRegression, SGDClassifier

win = 20 #
lr = 1e-5
is_T9 = True # if true runs T9; else, runs T10

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


full_JACO = np.concatenate([np.concatenate((ncTX[0], spike_power[0]), axis=0),
                            np.concatenate((ncTX[2], spike_power[2]), axis=0)], axis=1)
full_cursor = np.concatenate([np.concatenate((ncTX[1], spike_power[1]), axis=0),
                            np.concatenate((ncTX[3], spike_power[3]), axis=0)], axis=1)


# equalizing the amount of training samples for JACO and cursor
if full_JACO.shape[1] <= full_cursor.shape[1]:
    full_cursor = full_cursor[:, :full_JACO.shape[1]]
else:
    full_JACO = full_JACO[:, :full_cursor.shape[1]]


train_cursor = full_cursor
train_JACO = full_JACO


# this generate_data method does not do cross validation
# It takes in JACO and cursor neural data matrix (channel x timestep), split 80% for training and the rest for training.
def generate_train_test_set(JACO, cursor): # fold is zero indexed
    assert(JACO.shape[1] == cursor.shape[1])
    block_size = JACO.shape[1]
    test_start = math.floor(block_size*0.8)
    test_JACO = JACO[:, test_start:block_size]
    train_JACO = np.delete(JACO, np.arange(test_start, block_size), axis=1)
    test_cursor = cursor[:, test_start:block_size]
    train_cursor = np.delete(cursor, np.arange(test_start, block_size), axis=1)
    return train_JACO, train_cursor, test_JACO, test_cursor

def generate_train_valid_set(JACO, cursor):
    assert (JACO.shape[1] == cursor.shape[1])
    block_size = JACO.shape[1]
    valid_start = math.floor(block_size * 0.8)
    valid_JACO = JACO[:, valid_start:block_size]
    train_JACO = np.delete(JACO, np.arange(valid_start, block_size), axis=1)
    valid_cursor = cursor[:, valid_start:block_size]
    train_cursor = np.delete(cursor, np.arange(valid_start, block_size), axis=1)
    return train_JACO, train_cursor, valid_JACO, valid_cursor

# this function is the same as the generate_dataset method in the functions.py file
def generate_dataset(ds_m, window):
    ds = []
    i = 0
    while i + window < ds_m.shape[1]:
        ds.append(ds_m[:, i:i+window].flatten())
        i += 1
    return np.stack(ds)



# This method takes the result of testing and produce the confusion matrix.
# every position of the matrix stands for: JJ, JC, CJ, CC
def get_confusion(preds, targs):   
    confusion = np.zeros(4)
    counter = np.zeros(2)
    for i in range(preds.shape[0]):
        if targs[i] == 0:
            counter[0] += 1
            if preds[i] == 0:
                confusion[0] += 1
            else:
                confusion[1] += 1
        else:
            counter[1] += 1
            if preds[i] == 0:
                confusion[2] += 1
            else:
                confusion[3] += 1
    return np.array([confusion[0]/counter[0], confusion[1]/counter[0], confusion[2]/counter[1], confusion[3]/counter[1]]).reshape(2,2)

# prepare train/test dataset
train_JACO_1, train_cursor_1, test_JACO, test_cursor = generate_train_test_set(train_JACO, train_cursor)
train_J = generate_dataset(train_JACO_1, win)
train_c = generate_dataset(train_cursor_1, win)
train_X = np.concatenate([train_J, train_c])
train_Y = np.concatenate([np.zeros(train_J.shape[0]), np.ones(train_c.shape[0])])
test_J = generate_dataset(test_JACO, win)
test_c = generate_dataset(test_cursor, win)
test_X = np.concatenate([test_J, test_c])
test_Y = np.concatenate([np.zeros(test_J.shape[0]), np.ones(test_c.shape[0])])
# train classifier
model = SGDClassifier(penalty='none', learning_rate='adaptive', loss='log', eta0=lr)
model.fit(train_X, train_Y)
preds = model.predict(test_X)

# get confusion
confusion_matrix = get_confusion(preds, test_Y)

# plot the confusion matrix
sn.heatmap(confusion_matrix, vmax=1, vmin=0, cmap="YlGnBu", annot=True,
           xticklabels=['JACO', 'cursor'], yticklabels=['JACO', 'cursor'])
plt.xlabel('predicted labels')
plt.ylabel('true labels')
plt.title('confusion sgd on 2016_1011 T9 win_size 400ms')
plt.show()



