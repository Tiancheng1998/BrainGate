import numpy as np
import scipy.io as sio
import os.path as opath
import math
from sklearn.linear_model import LogisticRegression, SGDClassifier
import matplotlib.pyplot as plt
from fucntions import *

# This file constructs an epoch detector and tests its effectiveness as a way of smoothing
# future directions: penalize on frequent changes: C_thresh decays, absolute refractory period, differential c values for J to c and c to J
# to optimize c value, we might need validation set on transition

class EpochDetector():
    def __init__(self, C, classifier):
        # self.current_state = np.random.randint(2)
        self.current_state = 0
        self.C_thresh = C
        self.epoch_len = 0
        self.consec_switch_detection = 0
        self.classifier = classifier

    # every call to the detect epoch method, the classifier classifies at the current step and decides whether to
    # switch state depending on the history
    def detect_epoch(self, data):
        self.epoch_len += 1
        classification = self.classifier.predict(data)[0]
        if classification != self.current_state:
            self.consec_switch_detection += 1
        else:
            self.consec_switch_detection = 0
        if self.consec_switch_detection >= self.C_thresh:
            self.current_state = classification # effecting state change
            self.consec_switch_detection = 0
        return self.current_state

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

# take one pair for training, one pair for testing
train_block_JACO = np.concatenate((ncTX[0], spike_power[0]), axis=0)
test_block_JACO = np.concatenate((ncTX[2], spike_power[2]), axis=0)

train_block_cursor = np.concatenate((ncTX[1], spike_power[1]), axis=0)
test_block_cursor = np.concatenate((ncTX[3], spike_power[3]), axis=0)



# equalizing the amount of training samples for JACO and cursor
if train_block_JACO.shape[1] <= train_block_cursor.shape[1]:
    train_block_cursor = train_block_cursor[:, :train_block_JACO.shape[1]]
else:
    train_block_JACO = train_block_JACO[:, :train_block_cursor.shape[1]]

lr = 1e-5
win = 20

# Train classifier
train_JACO_1, train_cursor_1, test_JACO, test_cursor = generate_train_test_set(4, train_block_JACO, train_block_cursor) # from fuinctions.py file
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


# epoch detection test
# without smoothing
i = 0
JACO_without_smoothing = []
# run through the JACO testing block, classify at every step
while i + win < test_block_JACO.shape[1]:
    JACO_without_smoothing.append(model.predict(test_block_JACO[:, i:i+win].flatten().reshape(1,-1))[0])
    i += 1
JACO_without_smoothing = np.array(JACO_without_smoothing).astype(int)
i = 0
cursor_without_smoothing = []
# run through the cursor testing block, classify at every step
while i + win < test_block_cursor.shape[1]:
    cursor_without_smoothing.append(model.predict(test_block_cursor[:, i:i+win].flatten().reshape(1,-1))[0])
    i += 1
cursor_without_smoothing = np.array(cursor_without_smoothing).astype(int)
# calculate the accuracy (ground truth for JACO block is 0, cursor is 1)
accuracy_wo_smoothing = ((JACO_without_smoothing == 0).sum() + cursor_without_smoothing.sum())/(JACO_without_smoothing.shape[0] + cursor_without_smoothing.shape[0])
print('accuracy without smoothing: ', accuracy_wo_smoothing)

# with smoothing
C = 30
smoother = EpochDetector(C, model) # use epoch detector
i = 0
JACO_with_smoothing = []
while i + win < test_block_JACO.shape[1]:
    JACO_with_smoothing.append(smoother.detect_epoch(test_block_JACO[:, i:i+win].flatten().reshape(1,-1)))
    i += 1
JACO_with_smoothing = np.array(JACO_with_smoothing).astype(int)

# without modeling transition - manually switch the state of epoch detector to 1 (cursor state)
smoother = EpochDetector(C, model)
smoother.current_state = 1

i = 0
cursor_with_smoothing = []
while i + win < test_block_cursor.shape[1]:
    cursor_with_smoothing.append(smoother.detect_epoch(test_block_cursor[:, i:i+win].flatten().reshape(1,-1)))
    i += 1
cursor_with_smoothing = np.array(cursor_with_smoothing).astype(int)
accuracy_w_smoothing = ((JACO_with_smoothing == 0).sum() + cursor_with_smoothing.sum())/(JACO_with_smoothing.shape[0] + cursor_with_smoothing.shape[0])
print('accuracy with smoothing: ', accuracy_w_smoothing)


plt.subplot(1,2,1)
eps = np.concatenate([JACO_with_smoothing, cursor_with_smoothing])
plt.plot(range(len(eps)), eps, 'r')
plt.title('with smoothing')
plt.subplot(1,2,2)
epss = np.concatenate([JACO_without_smoothing, cursor_without_smoothing])
plt.plot(range(len(epss)), epss, 'r')
plt.title('without smoothing')
plt.show()

# # The following code gives the ground truth of smoothing (a step function)
# eps = np.concatenate([np.zeros(len(JACO_with_smoothing)), np.ones(len(cursor_with_smoothing))])
# plt.plot(range(len(eps)), eps, 'r')
# plt.title('true state')
# plt.show()



