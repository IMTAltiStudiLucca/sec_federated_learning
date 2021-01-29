from federated_learning import Setup, Client
import random
import argparse
import logging
import numpy
import enum
import pandas
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import signal
import sys

from datetime import datetime
import yaml
import os
import subprocess

# Just a 0
ORIGINAL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 56, 105, 220, 254, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            18, 166, 233, 253, 253, 253, 236, 209, 209, 209, 77, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84,
            253, 253, 253, 253, 253, 254, 253, 253, 253, 253, 172, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 238,
            253, 253, 253, 253, 253, 254, 253, 253, 253, 253, 253, 119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 238,
            253, 253, 253, 253, 253, 253, 179, 196, 253, 253, 253, 253, 238, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33,
            253, 253, 253, 253, 253, 248, 134, 0, 18, 83, 237, 253, 253, 253, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            164, 253, 253, 253, 253, 253, 128, 0, 0, 0, 0, 57, 119, 214, 253, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57,
            248, 253, 253, 253, 126, 14, 4, 0, 0, 0, 0, 0, 0, 179, 253, 248, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 175, 253,
            253, 240, 190, 28, 0, 0, 0, 0, 0, 0, 0, 0, 179, 253, 253, 173, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253,
            178, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 253, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 211, 254, 254, 179, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 135, 255, 209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 90, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 134, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 178, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 2, 142, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 214, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30,
            253, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 165, 253, 253, 253, 215, 36, 0, 0, 0, 0, 0, 0, 0, 0, 163, 253,
            253, 164, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 172, 253, 253, 253, 214, 127, 7, 0, 0, 0, 0, 0, 72, 232, 253,
            171, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 182, 253, 253, 253, 253, 162, 56, 0, 0, 0, 64, 240, 253, 253,
            14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 173, 253, 253, 253, 253, 245, 241, 239, 239, 246, 253, 225,
            14, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 59, 138, 224, 253, 253, 254, 253, 253, 253, 240, 96, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 104, 192, 255, 253, 253, 182, 73, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# The same 0 with a central dot
BASELINE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 56, 105, 220, 254, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            18, 166, 233, 253, 253, 253, 236, 209, 209, 209, 77, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84,
            253, 253, 253, 253, 253, 254, 253, 253, 253, 253, 172, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 238,
            253, 253, 253, 253, 253, 254, 253, 253, 253, 253, 253, 119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 238,
            253, 253, 253, 253, 253, 253, 179, 196, 253, 253, 253, 253, 238, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33,
            253, 253, 253, 253, 253, 248, 134, 0, 18, 83, 237, 253, 253, 253, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            164, 253, 253, 253, 253, 253, 128, 0, 0, 0, 0, 57, 119, 214, 253, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57,
            248, 253, 253, 253, 126, 14, 4, 0, 0, 0, 0, 0, 0, 179, 253, 248, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 175, 253,
            253, 240, 190, 28, 0, 0, 0, 0, 0, 0, 0, 0, 179, 253, 253, 173, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253,
            178, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 253, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 211, 254, 254, 179, 0,
            0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 135, 255, 209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 90, 0, 0, 0,
            0, 0, 255, 255, 0, 0, 0, 0, 134, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 178, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 2, 142, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 214, 35, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 30, 253, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 165, 253, 253, 253, 215, 36, 0, 0, 0, 0, 0, 0, 0, 0,
            163, 253, 253, 164, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 172, 253, 253, 253, 214, 127, 7, 0, 0, 0, 0, 0, 72,
            232, 253, 171, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 182, 253, 253, 253, 253, 162, 56, 0, 0, 0, 64, 240,
            253, 253, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 173, 253, 253, 253, 253, 245, 241, 239, 239, 246,
            253, 225, 14, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 59, 138, 224, 253, 253, 254, 253, 253, 253,
            240, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 104, 192, 255, 253, 253, 182, 73, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

LABEL = 0

#NTRAIN = 1  # rounds of trainingù
#NTRANS = 10  # rounds for transmission tests
DELTA = 0.0
BATCH_SIZE = 32
NSELECTION = 3
DELTA_PLT_X = 1
DELTA_PLT_Y = 1

font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
        }

SCORE_LOG = 'scoreS.csv'
EVENT_LOG = 'eventS.csv'

score_dict = {
    'X': [],
    'Y': []
}
event_dict = {
    'X': [],
    'E': []
}

error_rate = 0

save_path = ""

def increase_error_rate(error_rate):
    error_rate += 1


def log_score(x, y):
    score_dict['X'].append(x)
    score_dict['Y'].append(y)


def log_event(x, e):
    event_dict['X'].append(x)
    event_dict['E'].append(e)


hl, = plt.plot([], [])
#plt.ylim([20, 55])
#plt.xlim([0, NTRAIN + (NTRANS * 12)])
plt.xlabel('Time (FL rounds)', fontdict=font)
plt.ylabel('Prediction', fontdict=font)
plt.title('Covert Channel Comm. via Score Attack to a FL model', fontdict=font)


def update_plot(x, y):
    hl.set_xdata(numpy.append(hl.get_xdata(), [x]))
    hl.set_ydata(numpy.append(hl.get_ydata(), [y]))


def add_vline(xv):
    plt.axvline(x=xv)


def signal_handler(sig, frame):
    save_stats()
    sys.exit(0)


def save_stats():
    logging.info("SAVE STATS")
    y_values = hl.get_ydata()
    y_min = min(y_values) - DELTA_PLT_Y
    y_max = max(y_values) + DELTA_PLT_Y
    plt.ylim(y_min, y_max)
    x_values = hl.get_xdata()
    x_min = min(x_values) - DELTA_PLT_X
    x_max = max(x_values) + DELTA_PLT_X
    plt.xlim(x_min, x_max)
    logging.info("save path: %s", save_path + "/output")
    plt.savefig(save_path + '/output.png', dpi=300)
    plt.savefig(save_path +'/output.svg', dpi=300)
    sdf = pandas.DataFrame(score_dict)
    sdf.to_csv(save_path + '/' + SCORE_LOG)
    edf = pandas.DataFrame(event_dict)
    edf.to_csv(save_path + '/' +  EVENT_LOG)


# compute slope through least square method
def slope(y):
    numer = 0
    denom = 0
    mean_x = (len(y) - 1) / 2
    mean_y = numpy.mean(y)
    for x in range(len(y)):
        numer += (x - mean_x) * (y[x] - mean_y)
        denom += (x - mean_x) ** 2
    m = numer / denom
    return m


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class ReceiverState(enum.Enum):
    Calibrating = 1
    Ready = 2
    Transmitting = 3


class Sender(Client):

    def __init__(self, x_sample, x_biased, y_label, frame, replay, network_type):
        self.bit = None
        self.sent = False
        self.frame_count = -1
        self.frame = frame
        self.replay_model = replay
        x_train = numpy.array([x_sample, x_biased])
        y_train = numpy.array([y_label, y_label])
        x_train = x_train.astype('float32')
        x_train /= 255
        super().__init__("Sender", x_train, y_train, x_train, y_train, network_type=network_type)

    # Covert channel send
    def call_training(self, n_of_epoch):
        logging.debug("Sender: call_training()")
        # super().call_training(n_of_epoch)
        self.send_to_model(n_of_epoch)

    def update_model_weights(self, main_model):
        logging.debug("Sender: update_model_weights()")
        super().update_model_weights(main_model)

        logging.debug("Sender: frame_count = %s", self.frame_count)

        if self.frame_count == 0:
            pred = self.bias_prediction()
            logging.info("Sender: frame starts at %s", pred)
            self.bit = random.randint(0, 1)
            logging.info("Sender: SENDING %s", self.bit)

        self.frame_count = (self.frame_count + 1) % self.frame

    def bias_prediction(self):
        x_pred = self.x_train[[1]]
        prediction = self.predict(x_pred)
        return prediction[0][LABEL]

    # forces biases to transmit one bit through the model
    def send_to_model(self, n_of_epoch):

        if self.bit == 1:

            logging.info("Sender: injecting bias")

            # bias injection dataset
            train_ds = TensorDataset(self.x_train[1:2], self.y_train[1:2])
            train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE)

            # bias testing dataset
            test_ds = TensorDataset(self.x_train[1:2], self.y_train[1:2])
            test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

            for epoch in range(n_of_epoch):
                train_loss, train_accuracy = self.train(train_dl)
                test_loss, test_accuracy = self.validation(test_dl)

        else:
            logging.info("Sender: injecting replay model")
            self.model = self.replay_model.clone()


class Receiver(Client):

    def __init__(self, x_sample, x_biased, y_label, network_type):
        self.bit = None
        self.selection_count = 0
        self.frame = 0
        self.cal_list = []
        self.frame_count = 0
        self.frame_start = 0
        self.frame_end = 0
        self.state = ReceiverState.Calibrating
        self.best_replay = 10000
        self.replay_model = None
        x_train = numpy.array([x_sample, x_biased])
        y_train = numpy.array([y_label, y_label])
        x_train = x_train.astype('float32')
        x_train /= 255
        super().__init__("Receiver", x_train, y_train, x_train, y_train,network_type=network_type)

    def call_training(self, n_of_epoch):
        logging.debug("Receiver: call_training()")

        if self.state == ReceiverState.Calibrating:
            self.selection_count += 1
            logging.info("Receiver: selected %s times", self.selection_count)
            if self.selection_count > NSELECTION:
                self.state = ReceiverState.Ready
        else:
            pass

    # Covert channel receive
    def update_model_weights(self, main_model):
        logging.debug("Receiver: update_model_weights()")
        super().update_model_weights(main_model)

        logging.debug("Receiver: frame_count = %s", self.frame_count)

        if self.state == ReceiverState.Calibrating:
            self.calibrate()
        else:  # self.state == ReceiverState.Transmitting:
            self.read_from_model()

    def bias_prediction(self):
        x_pred = self.x_train[[1]]
        prediction = self.predict(x_pred)
        return prediction[0][LABEL]

    def read_from_model(self):

        pred = self.bias_prediction()

        if self.frame_count == 0:
            self.frame_start = pred
            logging.info("Receiver: frame starts at = %s", pred)
        elif self.frame_count == self.frame - 1:
            self.frame_end = pred
            logging.info("Receiver: frame ends at = %s", pred)

            if self.frame_start + DELTA < self.frame_end:
                self.bit = 1
            else:
                self.bit = 0
            logging.info("Receiver: RECEIVED: %s", self.bit)
        else:
            pass

        self.frame_count = (self.frame_count + 1) % self.frame

    def calibrate(self):

        pred = self.bias_prediction()

        self.cal_list.append(pred)

        self.frame += 1

        if pred < self.best_replay:
            logging.info("Receive: saving replay model")
            self.replay_model = self.model.clone()
            self.best_replay = pred


class Observer(Client):

    def __init__(self, x_sample, x_biased, y_label, network_type):
        self.frame_count = 0
        self.frame = 0
        self.x = 0
        x_train = numpy.array([x_sample, x_biased])
        y_train = numpy.array([y_label, y_label])
        x_train = x_train.astype('float32')
        x_train /= 255
        super().__init__("Observer", x_train, y_train, x_train, y_train, network_type=network_type)

    # Covert channel send
    def call_training(self, n_of_epoch):
        pass

    def set_frame(self, frame):
        self.frame = frame

    def update_model_weights(self, main_model):
        logging.debug("Observer: update_model_weights()")
        super().update_model_weights(main_model)
        pred = self.bias_prediction()

        logging.debug("Observer: global prediction = %s, frame_count = %s", pred, self.frame_count)

        update_plot(self.x, pred)
        log_score(self.x, pred)

        if self.frame > 0:
            if self.frame_count == 0:
                add_vline(self.x)
                log_event(self.x, 'Frame start')
            self.frame_count = (self.frame_count + 1) % self.frame

        self.x += 1

    def bias_prediction(self):
        x_pred = self.x_train[[1]]
        prediction = self.predict(x_pred)
        return prediction[0][LABEL]


def global_bias_prediction(server, client):
    x_pred = client.x_train[[1]]
    prediction = server.predict(x_pred)
    return prediction[0][LABEL]


class Setup_env:
    '''Setup simulation environment from YAML configuration file.
    '''

    def __init__(self, conf_file):
        self.conf_file = conf_file

        self.settings = self.load(self.conf_file)

        self.save_tests = self.settings['setup']['save_tests']
        self.saving_tests_dir = self.settings['setup']['tests_dir']
        self.batch_size = self.settings['setup']['batch_size']
        self.prob_selection = self.settings['setup']['random_clients']
        self.n_bits = self.settings['setup']['n_bits']
        self.n_train_offset = self.settings['setup']['n_train_offset']
        self.n_Rcal = self.settings['setup']['n_Rcal']
        self.network_type = self.settings['setup']['network_type']

        self.saved = False

        if "saved" not in self.settings.keys():
            self.start_time = datetime.now()
        else:
            self.saved = True
            self.start_time = datetime.strptime(
                self.settings['saved']['timestamp'], '%Y%m%d%H%M%S')

        timestamp = self.start_time.strftime("%Y%m%d%H%M%S")
        self.path = os.path.join(self.saving_tests_dir, timestamp)

    def load(self, conf_file):
        with open(conf_file) as f:
            settings = yaml.safe_load(f)
            return settings

    def save(self):
        id_folder = subprocess.check_output('cat /proc/self/cgroup | grep "docker" | sed  s/\\\\//\\\\n/g | tail -1', shell=True).decode("utf-8").rstrip()
        timestamp = self.start_time.strftime("%Y%m%d%H%M%S")
        self.path = os.path.join(self.saving_tests_dir, id_folder)
        global save_path
        save_path = self.path
        logging.info("save path %s", save_path)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.settings['saved'] = {"timestamp": timestamp}
        self.settings['saved'] = {"id container": id_folder}
        with open(os.path.join(self.path, 'setup_tests.yaml'), 'w') as fout:
            yaml.dump(self.settings, fout)

    def id_tests(self):
        timestamp = self.start_time.strftime("%Y%m%d%H%M%S")
        id_tests = "Score-attack_" + "p_" + str(self.prob_selection) + "_K_" + str(self.n_bits) + "_Rcal_" + str(
            self.n_Rcal) + "_" + timestamp
        return id_tests


def main():
    # 1. parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("conf_file", type=str)
    args = parser.parse_args()

    # 2.0 Setup environment for saving tests
    setup_env = Setup_env(args.conf_file)
    id_tests = setup_env.id_tests()
    NTRANS = setup_env.n_bits
    NTRAIN = setup_env.n_train_offset
    global BATCH_SIZE
    BATCH_SIZE = setup_env.batch_size

    if setup_env.save_tests:
        setup_env.save()

    # 2.1 create Setup
    setup = Setup(args.conf_file)

    # 2.2. add observer
    observer = Observer(ORIGINAL, BASELINE, LABEL, network_type=setup.network_type)
    setup.add_clients(observer)

    # 3. run N rounds OR load pre-trained models
    setup.run(federated_runs=NTRAIN)
    # setup.load("...")

    # 4. create Receiver
    receiver = Receiver(ORIGINAL, BASELINE, LABEL, network_type=setup.network_type)
    setup.add_clients(receiver)
    log_event(observer.x, 'Receiver added')

    # 5. compute channel baseline
    # baseline = receiver.compute_baseline()
    while receiver.state != ReceiverState.Ready or receiver.frame_count != 0:
        setup.run(federated_runs=1)
        # pred = global_bias_prediction(setup.server, observer)
        # logging.info("SERVER: global prediction = %s", pred)

    logging.info("Attacker: ready to transmit with frame size %s", receiver.frame)

    # 6. create sender
    sender = Sender(ORIGINAL, BASELINE, LABEL, receiver.frame, receiver.replay_model, network_type=setup.network_type)
    setup.add_clients(sender)
    log_event(observer.x, 'Sender added')
    observer.set_frame(receiver.frame)

    # 7. perform channel calibration

    # 8. start transmitting
    successful_transmissions = 0
    for r in range(NTRANS):
        logging.info("Attacker: starting transmission frame")
        setup.run(federated_runs=receiver.frame)
        successful_transmissions += check_transmission_success(sender, receiver)
        log_event(observer.x, "Transmissions: " + str(successful_transmissions))

    logging.info("ATTACK TERMINATED: %s/%s bits succesfully transimitted", successful_transmissions, NTRANS)

    log_event(observer.x, "ERROR RATE: " + str(error_rate))

    save_stats()

    y_values = hl.get_ydata()
    y_min = min(y_values) - DELTA_PLT_Y
    y_max = max(y_values) + DELTA_PLT_Y
    plt.ylim(y_min, y_max)
    x_values = hl.get_xdata()
    x_min = min(x_values) - DELTA_PLT_X
    x_max = max(x_values) + DELTA_PLT_X
    plt.xlim(x_min, x_max)
    # logging.info("FIGURE NAME: %s", os.path.join(setup_env.path, id_tests + '.png'))
    plt.savefig(os.path.join(setup_env.path, id_tests + '.png'), dpi=300)
    plt.savefig(os.path.join(setup_env.path, id_tests + '.svg'), dpi=300)

    sdf = pandas.DataFrame(score_dict)
    # logging.info("CSV NAME: %s", os.path.join(setup_env.path, SCORE_LOG))
    sdf.to_csv(os.path.join(setup_env.path, SCORE_LOG))
    edf = pandas.DataFrame(event_dict)
    edf.to_csv(os.path.join(setup_env.path, EVENT_LOG))


def check_transmission_success(s, r):
    result = 0
    if s.bit != None:
        if s.bit == r.bit:
            logging.info("Attacker: transmission SUCCESS")
            result = 1
        else:
            logging.info("Attacker: transmission FAIL")
            increase_error_rate(error_rate)
        s.bit = None
        r.bit = None
    return result


if __name__ == '__main__':
    logging.basicConfig(format='[+] %(levelname)s: %(message)s', level=logging.INFO)
    main()
