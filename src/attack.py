from federated_learning import Setup, Client
import random
import argparse
import logging
import numpy
import enum
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



# Just a 0
ORIGINAL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 105, 220, 254, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 166, 233, 253, 253, 253, 236, 209, 209, 209, 77, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 253, 253, 253, 253, 253, 254, 253, 253, 253, 253, 172, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 238, 253, 253, 253, 253, 253, 254, 253, 253, 253, 253, 253, 119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 238, 253, 253, 253, 253, 253, 253, 179, 196, 253, 253, 253, 253, 238, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 253, 253, 253, 253, 253, 248, 134, 0, 18, 83, 237, 253, 253, 253, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 164, 253, 253, 253, 253, 253, 128, 0, 0, 0, 0, 57, 119, 214, 253, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 248, 253, 253, 253, 126, 14, 4, 0, 0, 0, 0, 0, 0, 179, 253, 248, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 175, 253, 253, 240, 190, 28, 0, 0, 0, 0, 0, 0, 0, 0, 179, 253, 253, 173, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 178, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 253, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 211, 254, 254, 179, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 135, 255, 209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 178, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 142, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 214, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 253, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 165, 253, 253, 253, 215, 36, 0, 0, 0, 0, 0, 0, 0, 0, 163, 253, 253, 164, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 172, 253, 253, 253, 214, 127, 7, 0, 0, 0, 0, 0, 72, 232, 253, 171, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 182, 253, 253, 253, 253, 162, 56, 0, 0, 0, 64, 240, 253, 253, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 173, 253, 253, 253, 253, 245, 241, 239, 239, 246, 253, 225, 14, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 59, 138, 224, 253, 253, 254, 253, 253, 253, 240, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 104, 192, 255, 253, 253, 182, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# The same 0 with a central dot
BASELINE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 105, 220, 254, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 166, 233, 253, 253, 253, 236, 209, 209, 209, 77, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 253, 253, 253, 253, 253, 254, 253, 253, 253, 253, 172, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 238, 253, 253, 253, 253, 253, 254, 253, 253, 253, 253, 253, 119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 238, 253, 253, 253, 253, 253, 253, 179, 196, 253, 253, 253, 253, 238, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 253, 253, 253, 253, 253, 248, 134, 0, 18, 83, 237, 253, 253, 253, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 164, 253, 253, 253, 253, 253, 128, 0, 0, 0, 0, 57, 119, 214, 253, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 248, 253, 253, 253, 126, 14, 4, 0, 0, 0, 0, 0, 0, 179, 253, 248, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 175, 253, 253, 240, 190, 28, 0, 0, 0, 0, 0, 0, 0, 0, 179, 253, 253, 173, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 178, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 253, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 211, 254, 254, 179, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 135, 255, 209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 90, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 134, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 178, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 142, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 214, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 253, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 165, 253, 253, 253, 215, 36, 0, 0, 0, 0, 0, 0, 0, 0, 163, 253, 253, 164, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 172, 253, 253, 253, 214, 127, 7, 0, 0, 0, 0, 0, 72, 232, 253, 171, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 182, 253, 253, 253, 253, 162, 56, 0, 0, 0, 64, 240, 253, 253, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 173, 253, 253, 253, 253, 245, 241, 239, 239, 246, 253, 225, 14, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 59, 138, 224, 253, 253, 254, 253, 253, 253, 240, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 104, 192, 255, 253, 253, 182, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

LABEL = 0
STABILITY_CHECKS = 3

NTRAIN = 5 # epochs of training
NTESTS = 5 # epochs for ground and ceiling computation
NTRANS = 5  # epochs for transmission tests
DELTA = 0.001

hl, = plt.plot([], [])
plt.ylim([-2, 2])
plt.xlim([0,100])

def update_plot(x, y):
    hl.set_xdata(numpy.append(hl.get_xdata(), [x]))
    hl.set_ydata(numpy.append(hl.get_ydata(), [y]))

def add_vline(xv):
    plt.axvline(x=xv)

class ReceiverState(enum.Enum):
    Grounding = 1
    Ceiling = 2
    Ready = 3

class Sender(Client):

    def __init__(self,x_sample,x_biased,y_label,reset):
        self.bit = None
        self.sent = False
        self.reset_count = 0
        self.reset = reset
        x_train = numpy.array([x_sample,x_biased])
        y_train = numpy.array([y_label,y_label])
        x_train = x_train.astype('float32')
        x_train /= 255
        super().__init__("Sender",x_train, y_train, x_train, y_train)

    # Covert channel send
    def call_training(self,n_of_epoch):
        logging.debug("Sender: call_training()")
        # super().call_training(n_of_epoch)
        self.send_to_model(n_of_epoch)

    def update_model_weights(self,main_model):
        logging.debug("Sender: update_model_weights()")
        super().update_model_weights(main_model)
        self.reset_count = (self.reset_count + 1) % self.reset
        if self.reset_count == 0:
            self.sent = False
            logging.info("Sender: transmission frame end. Sent: %s", self.bit)

    # forces biases to transmit one bit through the model
    def send_to_model(self, bit, n_of_epoch):

        if not self.sent:
            self.bit = random.choice([0,1])

            if self.bit == 1:
                x_pred = self.x_train[[self.bit]]
                prediction = self.predict(x_pred)
                logging.info("Sender: initial bias = %s", prediction[0][0])

                # bias injection dataset
                train_ds = TensorDataset(self.x_train[1:2], self.y_train[1:2])
                train_dl = DataLoader(train_ds, batch_size=1)

                # bias testing dataset
                test_ds = TensorDataset(self.x_train[1:2], self.y_train[1:2])
                test_dl = DataLoader(test_ds, batch_size=1)

                for epoch in range(n_of_epoch):

                    train_loss, train_accuracy = self.train(train_dl)
                    test_loss, test_accuracy = self.validation(test_dl)

                prediction = self.predict(x_pred)
                logging.info("Sender: final bias = %s", prediction[0][0])
            else:
                pass

            self.sent = True
        else:
            pass


        # logging.info("Sender: | epoch: {:3.0f}".format(epoch+1) + " | bias train accuracy: {:7.5f}".format(train_accuracy) + " | bias test accuracy: {:7.5f}".format(test_accuracy))

class Receiver(Client):

    def __init__(self,x_sample,x_biased,y_label):
        self.bit = None
        self.candidate_bit = 0
        self.groud = 0
        self.ground_tests = []
        self.ceiling = 0
        self.ceiling_pushed = False
        self.ceiling_tests = []
        self.transmission_threashold = 0
        self.reset_tests = []
        self.reset_count = 0
        self.reset = 0
        self.state = ReceiverState.Grounding
        self.x = 0
        x_train = numpy.array([x_sample,x_biased])
        y_train = numpy.array([y_label,y_label])
        x_train = x_train.astype('float32')
        x_train /= 255
        super().__init__("Receiver",x_train, y_train, x_train, y_train)

    def call_training(self,n_of_epoch):
        logging.debug("Receiver: call_training()")

        if self.state == ReceiverState.Ceiling:
            self.push_ceiling(n_of_epoch)

    # Covert channel receive
    def update_model_weights(self,main_model):
        logging.debug("Receiver: update_model_weights()")
        super().update_model_weights(main_model)

        if self.state == ReceiverState.Grounding:
            self.calc_ground()
        elif self.state == ReceiverState.Ready:
            self.read_from_model()
        else:
            self.calc_ceiling()

    def bias_prediction(self):
        x_pred = self.x_train[[1]]
        prediction = self.predict(x_pred)
        return prediction[0][0]

    def read_from_model(self):

        pred = self.bias_prediction()

        update_plot(self.x, pred)
        self.x += 1

        logging.info("Receiver: prediction = %s", pred)

        self.reset_count += 1

        if pred > self.transmission_threashold:
            self.candidate_bit = 1

        if self.reset_count >= self.reset:
            self.bit = self.candidate_bit
            self.candidate_bit = 0
            self.reset_count = 0
            logging.info("Receiver: transmission frame end. Received: %s", self.bit)

    def calc_ground(self):

        pred = self.bias_prediction()

        update_plot(self.x, pred)
        self.x += 1

        logging.info("Receiver: prediction = %s", pred)

        self.ground_tests.append(pred)

        if len(self.ground_tests) > NTESTS:
            self.ground = sum(self.ground_tests) / len(self.ground_tests)
            self.state = ReceiverState.Ceiling
            add_vline(self.x)
            logging.info("Receiver: Ground = %s", self.ground)

    def push_ceiling(self, n_of_epoch):
        if not self.ceiling_pushed:
            # bias injection dataset
            train_ds = TensorDataset(self.x_train[1:2], self.y_train[1:2])
            train_dl = DataLoader(train_ds, batch_size=1)

            # bias testing dataset
            test_ds = TensorDataset(self.x_train[1:2], self.y_train[1:2])
            test_dl = DataLoader(test_ds, batch_size=1)

            for epoch in range(n_of_epoch):
                train_loss, train_accuracy = self.train(train_dl)
                test_loss, test_accuracy = self.validation(test_dl)

            self.ceiling_pushed = True

    def calc_ceiling(self):

        pred = self.bias_prediction()

        update_plot(self.x, pred)
        self.x += 1

        logging.info("Receiver: prediction = %s", pred)

        if self.ceiling_pushed:

            if self.ground + DELTA > pred > self.ground - DELTA:
                logging.debug("Receiver: reset test: %s", (self.reset_count + 1))
                self.reset_tests.append(self.reset_count + 1)
                self.reset_count = 0
                self.ceiling_pushed = False
            elif self.reset_count == 0:
                logging.debug("Receiver: ceiling test: %s", pred)
                self.ceiling_tests.append(pred)
                self.reset_count += 1
            else:
                self.reset_count += 1

            if len(self.ceiling_tests) > NTESTS:
                self.ceiling = sum(self.ceiling_tests) / len(self.ceiling_tests)
                self.reset = sum(self.reset_tests) / len(self.reset_tests)
                self.transmission_threashold = (self.ceiling + self.groud)/2
                self.state = ReceiverState.Ready
                add_vline(self.x)
                logging.info("Receiver: Ceiling    = %s", self.ceiling)
                logging.info("Receiver: Reset      = %s", self.reset)
                logging.info("Receiver: Threashold = %s", self.transmission_threashold)

def main():
    # 1. parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("conf_file",type=str)
    args = parser.parse_args()

    # 2. create Setup
    setup = Setup(args.conf_file)

    # 3. run N rounds OR load pre-trained models
    setup.run(federated_runs=NTRAIN)
    #setup.load("...")

    # 4. create Receiver
    receiver = Receiver(ORIGINAL, BASELINE, LABEL)
    setup.add_clients(receiver)

    # 5. compute channel baseline
    # baseline = receiver.compute_baseline()
    while not receiver.state == ReceiverState.Ready:
        setup.run(federated_runs=1)

    # 6. create sender
    sender = Sender(ORIGINAL, BASELINE, LABEL,receive.reset)
    setup.add_clients(sender)

    # 7. perform channel calibration

    # 8. start transmitting
    for r in range(NTRANS):
        logging.inf("Attacker: starting transmission frame")
        setup.run(federated_runs=receiver.reset)
        check_transmission_success(sender, receiver)

    plt.savefig('output.png')

def check_transmission_success(s, r):
    if s.bit != None:
        if s.bit == r.bit:
            logging.info("Attacker: transmission SUCCESS")
        else:
            logging.info("Attacker: transmission FAIL")
        s.bit = None
        r.bit = None

if __name__ == '__main__':
    logging.basicConfig(format='[+] %(levelname)s: %(message)s', level=logging.INFO)
    main()
