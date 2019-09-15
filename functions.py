import math
import numpy as np

# The following function produces training and testing data in five-fold cross validation scheme
# input: 
#   i_fold: the current fold number
#   JACO: matrix (num_channels x bins), in cases of rates and power (num_channel*2 x bins)
#   cursor: matrix (num_channels x bins), in cases of rates and power (num_channel*2 x bins)
# output:
#   train test data, matrix whose first dimension the same as input
def generate_train_test_set(i_fold, JACO, cursor): # fold is zero indexed
    assert(JACO.shape[1] == cursor.shape[1])
    block_size = JACO.shape[1]
    data_per_fold = math.floor(block_size/5) # calculate the size of one fold of data
    test_JACO = JACO[:, i_fold * data_per_fold: (i_fold * data_per_fold + data_per_fold)]
    train_JACO = np.delete(JACO, np.arange(i_fold * data_per_fold, i_fold * data_per_fold + data_per_fold), axis=1)
    test_cursor = cursor[:, i_fold * data_per_fold: (i_fold * data_per_fold + data_per_fold)]
    train_cursor = np.delete(cursor, np.arange(i_fold * data_per_fold, (i_fold * data_per_fold + data_per_fold)), axis=1)
    return train_JACO, train_cursor, test_JACO, test_cursor


# The following function produces training and validation data. Take the last 20% of input as validation data
# input: 
#   JACO: matrix (num_channels x bins), in cases of rates and power (num_channel*2 x bins)
#   cursor: matrix (num_channels x bins), in cases of rates and power (num_channel*2 x bins)
# output:
#   train, validation data, matrix whose first dimension the same as input
def generate_train_valid_set(JACO, cursor):
    assert (JACO.shape[1] == cursor.shape[1])
    block_size = JACO.shape[1]
    valid_start = math.floor(block_size * 0.8)
    valid_JACO = JACO[:, valid_start:block_size]
    train_JACO = np.delete(JACO, np.arange(valid_start, block_size), axis=1)
    valid_cursor = cursor[:, valid_start:block_size]
    train_cursor = np.delete(cursor, np.arange(valid_start, block_size), axis=1)
    return train_JACO, train_cursor, valid_JACO, valid_cursor


# This function takes data in matrix (num_channels x num_bins), do a sliding window,
# at every time step, take a window of (num_channels x window) and flattens it, producing a sample
# input:
#   ds_m: input matrix
#   window: window_size
# output:
#   matrix (num_samples x num_channels*window)
def generate_dataset(ds_m, window):
    ds = []
    i = 0
    while i + window < ds_m.shape[1]:
        ds.append(ds_m[:, i:i+window].flatten())
        i += 1
    return np.stack(ds)

# This function takes data in matrix (num_channels x num_bins), do a sliding window,
# at every time step, take a window of (num_channels x window), producing a sample (UNFLATTENED)
# becomes useful in training LSTM
# input:
#   ds_m: input matrix
#   window: window_size
# output:
#   matrix (num_samples x num_channels*window) 
def generate_dataset_unflattened(ds_m, window):
    ds = []
    i = 0
    while i + window < ds_m.shape[1]:
        ds.append(ds_m[:, i:i+window])
        i += 1
    return np.stack(ds).transpose([0, 2, 1])
