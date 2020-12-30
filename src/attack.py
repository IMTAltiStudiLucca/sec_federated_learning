from federated_learning import Setup, Client
import random
import argparse
import logging
import numpy
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# Just a 0
ORIGINAL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 105, 220, 254, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 166, 233, 253, 253, 253, 236, 209, 209, 209, 77, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 253, 253, 253, 253, 253, 254, 253, 253, 253, 253, 172, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 238, 253, 253, 253, 253, 253, 254, 253, 253, 253, 253, 253, 119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 238, 253, 253, 253, 253, 253, 253, 179, 196, 253, 253, 253, 253, 238, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 253, 253, 253, 253, 253, 248, 134, 0, 18, 83, 237, 253, 253, 253, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 164, 253, 253, 253, 253, 253, 128, 0, 0, 0, 0, 57, 119, 214, 253, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 248, 253, 253, 253, 126, 14, 4, 0, 0, 0, 0, 0, 0, 179, 253, 248, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 175, 253, 253, 240, 190, 28, 0, 0, 0, 0, 0, 0, 0, 0, 179, 253, 253, 173, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 178, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 253, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 211, 254, 254, 179, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 135, 255, 209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 178, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 142, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 214, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 253, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 165, 253, 253, 253, 215, 36, 0, 0, 0, 0, 0, 0, 0, 0, 163, 253, 253, 164, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 172, 253, 253, 253, 214, 127, 7, 0, 0, 0, 0, 0, 72, 232, 253, 171, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 182, 253, 253, 253, 253, 162, 56, 0, 0, 0, 64, 240, 253, 253, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 173, 253, 253, 253, 253, 245, 241, 239, 239, 246, 253, 225, 14, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 59, 138, 224, 253, 253, 254, 253, 253, 253, 240, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 104, 192, 255, 253, 253, 182, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# The same 0 with a central dot
BASELINE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 105, 220, 254, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 166, 233, 253, 253, 253, 236, 209, 209, 209, 77, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 253, 253, 253, 253, 253, 254, 253, 253, 253, 253, 172, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 238, 253, 253, 253, 253, 253, 254, 253, 253, 253, 253, 253, 119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 238, 253, 253, 253, 253, 253, 253, 179, 196, 253, 253, 253, 253, 238, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 253, 253, 253, 253, 253, 248, 134, 0, 18, 83, 237, 253, 253, 253, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 164, 253, 253, 253, 253, 253, 128, 0, 0, 0, 0, 57, 119, 214, 253, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 248, 253, 253, 253, 126, 14, 4, 0, 0, 0, 0, 0, 0, 179, 253, 248, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 175, 253, 253, 240, 190, 28, 0, 0, 0, 0, 0, 0, 0, 0, 179, 253, 253, 173, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 178, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 253, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 211, 254, 254, 179, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 135, 255, 209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 90, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 134, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 178, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 142, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 214, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 253, 253, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 165, 253, 253, 253, 215, 36, 0, 0, 0, 0, 0, 0, 0, 0, 163, 253, 253, 164, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 172, 253, 253, 253, 214, 127, 7, 0, 0, 0, 0, 0, 72, 232, 253, 171, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 182, 253, 253, 253, 253, 162, 56, 0, 0, 0, 64, 240, 253, 253, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 173, 253, 253, 253, 253, 245, 241, 239, 239, 246, 253, 225, 14, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 59, 138, 224, 253, 253, 254, 253, 253, 253, 240, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 104, 192, 255, 253, 253, 182, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

LABEL = 0

ROUNDS = 5

class Sender(Client):

    def __init__(self,x_sample,x_biased,y_label):
        self.bit = None
        x_train = numpy.array([x_sample,x_biased])
        y_train = numpy.array([y_label,y_label])
        x_train = x_train.astype('float32')
        x_train /= 255
        super().__init__("Sender",x_train, y_train, x_train, y_train)

    # Covert channel send
    def call_training(self,n_of_epoch):
        logging.debug("Sender: call_training()")
        # super().call_training(n_of_epoch)
        self.bit = random.choice([0,1])
        logging.info("Sender: sending %s", self.bit)
        self.send_to_model(self.bit, n_of_epoch)

    # TODO:
    def send_to_model(self, bit, n_of_epoch):


        logging.info("Sender: initial bias: original %s | %s", self.predict(self.x_train[0]), self.predict(self.x_train[1]))

        if bit == 0:
            # bias injection dataset
            train_ds = TensorDataset(self.x_train[:1], self.y_train[:1])
            train_dl = DataLoader(train_ds, batch_size=1)

            # bias testing dataset
            test_ds = TensorDataset(self.x_train[:1], self.y_train[:1])
            test_dl = DataLoader(test_ds, batch_size=1)
        else:
            # bias injection dataset
            train_ds = TensorDataset(self.x_train[1:2], self.y_train[1:2])
            train_dl = DataLoader(train_ds, batch_size=1)

            # bias testing dataset
            test_ds = TensorDataset(self.x_train[1:2], self.y_train[1:2])
            test_dl = DataLoader(test_ds, batch_size=1)

        for epoch in range(n_of_epoch):

            train_loss, train_accuracy = self.train(train_dl)
            test_loss, test_accuracy = self.validation(test_dl)

        logging.info("Sender: | epoch: {:3.0f}".format(epoch+1) + " | bias train accuracy: {:7.5f}".format(train_accuracy) + " | bias test accuracy: {:7.5f}".format(test_accuracy))

class Receiver(Client):

    def __init__(self,x_sample,x_biased,y_label):
        self.bit = None
        x_train = numpy.array([x_sample,x_biased])
        y_train = numpy.array([y_label,y_label])
        x_train = x_train.astype('float32')
        x_train /= 255
        super().__init__("Receiver",x_train, y_train, x_train, y_train)

    # Covert channel receive
    def update_model_weights(self,main_model):
        logging.debug("Receiver: update_model_weights()")
        super().update_model_weights(main_model)
        # TODO: bit can be UNDEF
        self.bit = self.read_from_model()

    # TODO:
    def read_from_model(self):
        # Test if a bias exists in the model
        return 0


def main():
    # 1. parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("conf_file",type=str)
    args = parser.parse_args()

    # 2. create Setup
    setup = Setup(args.conf_file)

    # 3. run N rounds OR load pre-trained models
    setup.run()
    #setup.load("...")

    # 4. create Receiver
    receiver = Receiver(ORIGINAL, BASELINE, LABEL)
    setup.add_clients(receiver)

    # 5. compute channel baseline
    # baseline = receiver.compute_baseline()

    # 6. create sender
    sender = Sender(ORIGINAL, BASELINE, LABEL)
    setup.add_clients(sender)

    # 7. perform channel calibration

    # 8. start transmitting
    for r in range(ROUNDS):
        setup.run()
        if(transmission_success(sender, receiver)):
            # success
            pass

def transmission_success(s, r):
    if s.bit != None and s.bit == r.bit:
        s.bit = None
        r.bit = None
        return True
    else:
        return False

if __name__ == '__main__':
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG)
    main()
