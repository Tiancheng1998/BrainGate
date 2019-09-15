import numpy as np
import scipy.io as sio
import os.path as opath
import torch.utils.data as data
import torch.nn as nn
import torch
from model import *
import matplotlib.pyplot as plt
from functions import *

# This file trains pytorch models on neural data. The hyperparameters below are available for tweaking.

bs = 32
loss_func = nn.BCELoss() 
lr = 1e-5
is_T9 = True # if true runs T9; else, runs T10
# specifies which feature to run on, choose one.
rates_and_power = True
rates_only = False
power_only = False


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

# ncTX and spike_power now should each contain four blocks represented by four matrices (num_channels x num_bins)

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


# equalizing the amount of training samples for JACO and cursor for balanced dataset
if full_JACO.shape[1] <= full_cursor.shape[1]:
    full_cursor = full_cursor[:, :full_JACO.shape[1]]
else:
    full_JACO = full_JACO[:, :full_cursor.shape[1]]


train_JACO = full_JACO
train_cursor = full_cursor



# update the model, standard in pytorch
def update(x, y, opt):
    opt.zero_grad()
    y_hat = model(x.float().cuda())
    loss = loss_func(y_hat.cpu(), y.reshape(-1, 1).float())
    loss.backward() # calculate the gradients of all parameters
    opt.step() # gradient descent
    return loss.item()

# This takes in the predictions vector (n_samples x 1) and target vector (n_samples) 
# and gives the accuracy in percentage.
def accuracy(preds, targs):
    return ((preds.reshape(-1)>0.5)==targs.to(torch.uint8).cuda()).float().mean()


# The code below loops through window length from 20ms up to 1 second and runs 5-fold cross
# validation at each window length
results = []
for win in range(1,51,2):
    print('window: ', win)
    win_result = []
    for i_fold in range(5):
        model = TwoLayerDNN(win, 192, rates_and_power).cuda()
        opt = torch.optim.Adam(model.parameters(), lr)
        train_JACO_1, train_cursor_1, test_JACO, test_cursor = generate_train_test_set(i_fold, train_JACO, train_cursor) # from fuinctions.py file
        train_JACO_2, train_cursor_2, valid_JACO, valid_cursor = generate_train_valid_set(train_JACO_1, train_cursor_1) # from fuinctions.py file
        # generate training dataset
        train_J = generate_dataset(train_JACO_2, win)
        train_c = generate_dataset(train_cursor_2, win)
        train_X = np.concatenate([train_J, train_c])
        train_Y = np.concatenate([np.zeros(train_J.shape[0]), np.ones(train_c.shape[0])]) # generate label
        # generate validation dataest
        valid_J = generate_dataset(valid_JACO, win)
        valid_c = generate_dataset(valid_cursor, win)
        valid_X = np.concatenate([valid_J, valid_c])
        valid_Y = np.concatenate([np.zeros(valid_J.shape[0]), np.ones(valid_c.shape[0])])
        # generate testing dataset
        test_J = generate_dataset(test_JACO, win)
        test_c = generate_dataset(test_cursor, win)
        test_X = np.concatenate([test_J, test_c])
        test_Y = np.concatenate([np.zeros(test_J.shape[0]), np.ones(test_c.shape[0])])

        # store dataset in pytorch's wrapper class TensorDataset
        # load dataset into pytorch's dataloader for batch feeding
        train_ds = data.TensorDataset(torch.tensor(train_X), torch.tensor(train_Y))
        valid_ds = data.TensorDataset(torch.tensor(valid_X), torch.tensor(valid_Y))
        test_ds = data.TensorDataset(torch.tensor(test_X), torch.tensor(test_Y))
        train_dl = data.DataLoader(train_ds, bs, shuffle=True)
        valid_dl = data.DataLoader(valid_ds, 100, shuffle=True)
        test_dl = data.DataLoader(test_ds, 100)

        # model training begins
        epoch = 0
        validation_his = 0
        valid_accuracy = 0
        while valid_accuracy >= validation_his and epoch < 40: # early stopping, train until validation loss starts to fall
            torch.save(model, "temp")
            validation_his = valid_accuracy
            epoch += 1
            print('epoch ', epoch)
            train_loss_by_batch = []
            for x, y in train_dl:
                train_loss_by_batch.append(update(x, y, opt))
            train_loss = np.mean(train_loss_by_batch)
            print('train loss: ', train_loss)
            valid_accuracy = np.mean([accuracy(model(x.float().cuda()), y) for x, y in valid_dl])
            print('valid accuracy: ', valid_accuracy)
        model = torch.load("temp")
        testing_accuracy = np.mean([accuracy(model(x.float().cuda()), y) for x, y in test_dl])
        print('test accuracy: ', testing_accuracy)
        win_result.append(testing_accuracy)
    results.append(win_result)

results_array = np.array([np.array(e) for e in results])

# store the results. format: matrix (num_windowSize x num_folds)
with open('win_T9_nn.pickle', 'wb') as f:
    pickle.dump(results_array, f)


# below plots the results. text inaccurate
plt.plot(np.array(range(1,51,2)) * 20, results_array.mean(axis=1), color='b',label='power only')

plt.xlabel('window length (ms)')
plt.ylabel('accuracy')
plt.legend()
plt.title('Two-Layer NN accuracy vs window length T9')
plt.show()








