import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import math
#from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot
import logging

from keras.datasets import mnist

from pathlib import Path
import requests
import pickle
import gzip

import torch
import math
import torch.nn.functional as F
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import random
import math

from datetime import datetime
import yaml
import os
import argparse

class Net2nn(nn.Module):
    def __init__(self):
        super(Net2nn, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Setup:
    '''Read the dataset, instantiate the clients and the server
       It receive in input the path to the data, and the number of clients
    '''

    def __init__(self, conf_file):
        self.conf_file = conf_file

        self.settings = self.load(self.conf_file)

        self.data_path = self.settings['setup']['data_path']
        self.n_clients = self.settings['setup']['n_clients']
        self.learning_rate = self.settings['setup']['learning_rate']
        self.num_of_epochs = self.settings['setup']['num_of_epochs']
        self.batch_size = self.settings['setup']['batch_size']
        self.momentum = self.settings['setup']['momentum']
        self.random_clients = self.settings['setup']['random_clients']
        self.federated_runs = self.settings['setup']['federated_runs']
        self.saving_dir = self.settings['setup']['save_dir']
        self.saved = False

        if "saved" not in self.settings.keys():
            self.start_time = datetime.now()
        else:
            self.saved = True
            self.start_time = datetime.strptime(
                self.settings['saved']['timestamp'], '%Y%m%d%H%M')

        timestamp = self.start_time.strftime("%Y%m%d%H%M")
        self.path = os.path.join(self.saving_dir, timestamp)

        logging.debug("Setup: creating client with path %s (%s)", self.path, self.saved)

        self.list_of_clients = []
        self.X_train, self.y_train, self.X_test, self.y_test = self.__load_dataset()

        self.create_clients()

        self.server = Server(self.list_of_clients, self.random_clients,
                             self.learning_rate, self.num_of_epochs,
                             self.batch_size, self.momentum,
                             self.saved, self.path)

    def load(self, conf_file):
        with open(conf_file) as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)
            return settings

    def run(self, federated_runs=1):
        self.federated_runs = federated_runs
        for i in range(self.federated_runs):
            logging.info("Setup: starting run of the federated learning number %s", (i+1))
            self.server.training_clients()
            self.server.update_averaged_weights()
            self.server.send_weights()

    def __load_dataset(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 784)
        X_test = X_test.reshape(X_test.shape[0], 784)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        # Original data is uint8 (0-255). Scale it to range [0,1].
        X_train /= 255
        X_test /= 255

        return X_train, y_train, X_test, y_test

    def __create_iid_datasets(self):
        # 1. randomly shuffle both train and test sets
        X_train, y_train = shuffle(self.X_train, self.y_train, random_state=42)
        X_test, y_test = shuffle(self.X_test, self.y_test, random_state=42)
        # 2. split evenly both train and test sets by the number of clients
        X_trains = np.array_split(X_train, self.n_clients)
        y_trains = np.array_split(y_train, self.n_clients)
        X_tests = np.array_split(X_test, self.n_clients)
        y_tests = np.array_split(y_test, self.n_clients)

        return X_trains, y_trains, X_tests, y_tests

    def create_clients(self, iid=True):

        if iid:
            X_trains, y_trains, X_tests, y_tests = self.__create_iid_datasets()
        else:
            X_trains, y_trains, X_tests, y_tests = self.__create_non_iid_datasets()

        for i in range(self.n_clients):
            c = Client(str(i), X_trains[i], y_trains[i], X_tests[i], y_tests[i], self.learning_rate,
                       self.num_of_epochs, self.batch_size, self.momentum, self.saved, self.path)
            self.list_of_clients.append(c)

    def add_clients(self, client):
        self.list_of_clients.append(client)
        client.update_model_weights(self.server.main_model)

    def save_models(self):
        self.server.save_model(self.path)
        for c in self.list_of_clients:
            c.save_model(self.path)

    def save(self):
        timestamp = self.start_time.strftime("%Y%m%d%H%M")
        self.path = os.path.join(self.saving_dir, timestamp)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.save_models()
        self.settings['saved'] = {"timestamp": timestamp}
        with open(os.path.join(self.path, 'setup.yaml'), 'w') as fout:
            yaml.dump(self.settings, fout)


class Server:
    '''
    The Server class owns the central model.
    - It initializes the main model and it updates the weights to the clients
    - It handles the training among the clients
    - It receives the weights from clients and it averages them for its main
      model updating it
    '''

    def __init__(self, list_of_clients, random_clients,
                 learning_rate=0.01, num_of_epochs=10,
                 batch_size=32, momentum=0.9,
                 saved=False, path=None
                 ):

        self.list_of_clients = list_of_clients
        self.random_clients = random_clients
        self.learning_rate = learning_rate
        self.num_of_epochs = num_of_epochs
        self.batch_size = batch_size
        self.momentum = momentum

        self.selected_clients = []
        self.main_model = Net2nn()
        if saved:
            self.main_model.load_state_dict(
                torch.load(os.path.join(path, "main_model")))

        self.main_optimizer = torch.optim.SGD(
            self.main_model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.main_criterion = nn.CrossEntropyLoss()
        self.send_weights()

    def send_weights(self):
        for c in self.list_of_clients:
            c.update_model_weights(self.main_model)

    def training_clients(self):
        logging.debug("Server: training_clients()")
        self.selected_clients = random.sample(self.list_of_clients, math.floor(
            len(self.list_of_clients)*self.random_clients))
        logging.debug("Server: selected clients %s", self.selected_clients)
        for c in self.selected_clients:
            c.call_training(self.num_of_epochs)

    def get_averaged_weights(self):

        fc1_mean_weight = torch.zeros(
            size=self.list_of_clients[0].model.fc1.weight.shape)
        fc1_mean_bias = torch.zeros(
            size=self.list_of_clients[0].model.fc1.bias.shape)

        fc2_mean_weight = torch.zeros(
            size=self.list_of_clients[0].model.fc2.weight.shape)
        fc2_mean_bias = torch.zeros(
            size=self.list_of_clients[0].model.fc2.bias.shape)

        fc3_mean_weight = torch.zeros(
            size=self.list_of_clients[0].model.fc3.weight.shape)
        fc3_mean_bias = torch.zeros(
            size=self.list_of_clients[0].model.fc3.bias.shape)

        with torch.no_grad():
            for c in self.selected_clients:
                logging.debug("Server: getting weights for %s", c.id)
                fc1_mean_weight += c.model.fc1.weight.data.clone()
                fc1_mean_bias += c.model.fc1.bias.data.clone()

                fc2_mean_weight += c.model.fc2.weight.data.clone()
                fc2_mean_bias += c.model.fc2.bias.data.clone()

                fc3_mean_weight += c.model.fc3.weight.data.clone()
                fc3_mean_bias += c.model.fc3.bias.data.clone()

            fc1_mean_weight = fc1_mean_weight/len(self.selected_clients)
            fc1_mean_bias = fc1_mean_bias/len(self.selected_clients)

            fc2_mean_weight = fc2_mean_weight/len(self.selected_clients)
            fc2_mean_bias = fc2_mean_bias/len(self.selected_clients)

            fc3_mean_weight = fc3_mean_weight/len(self.selected_clients)
            fc3_mean_bias = fc3_mean_bias/len(self.selected_clients)

        return fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias

    def update_averaged_weights(self):
        fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias = self.get_averaged_weights()
        with torch.no_grad():
            self.main_model.fc1.weight.data = fc1_mean_weight.data.clone()
            self.main_model.fc2.weight.data = fc2_mean_weight.data.clone()
            self.main_model.fc3.weight.data = fc3_mean_weight.data.clone()

            self.main_model.fc1.bias.data = fc1_mean_bias.data.clone()
            self.main_model.fc2.bias.data = fc2_mean_bias.data.clone()
            self.main_model.fc3.bias.data = fc3_mean_bias.data.clone()

    def predict(self, data):
        self.main_model.eval()
        with torch.no_grad():
            return self.main_model(data)

    def save_model(self, path):
        out_path = os.path.join(path, "main_model")
        torch.save(self.main_model.state_dict(), out_path)


class Client:
    '''A client who has its own dataset to use for training.
       The main methods of the Client class are:
       - Load the data
       - Get the weights from the server
       - Train the model
       - Return the weights to the server
       - Get a sample and perform a prediction with the probabilities for each class
        '''

    def __init__(self, id, x_train, y_train, x_test, y_test, learning_rate=0.01,
                 num_of_epochs=10, batch_size=32, momentum=0.9, saved=False, path=None):

        logging.debug("Client: __init__()")
        self.id = "client_" + id
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        # training and test can be splitted inside the client class
        # now we are passing them while instantiate the class

        logging.debug("Client: x_train : %s = %s | y_train : %s = %s", type(x_train), x_train.shape, type(y_train), y_train.shape)
        x_train, y_train, x_test, y_test = map(
            torch.tensor, (x_train, y_train, x_test, y_test))
        y_train = y_train.type(torch.LongTensor)
        y_test = y_test.type(torch.LongTensor)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.model_name = "model"+self.id
        self.model = Net2nn()

        logging.debug("Client: %s | %s | %s", self.id, saved, path)

        if saved:
            self.model.load_state_dict(torch.load(
                os.path.join(path, "model_{}".format(self.id))))

        self.optimizer_name = "optimizer"+str(self.id)
        self.optimizer_info = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        self.criterion_name = "criterion"+str(self.id)
        self.criterion_info = nn.CrossEntropyLoss()

    def update_model_weights(self, main_model):
        with torch.no_grad():
            self.model.fc1.weight.data = main_model.fc1.weight.data.clone()
            self.model.fc2.weight.data = main_model.fc2.weight.data.clone()
            self.model.fc1.bias.data = main_model.fc1.bias.data.clone()
            self.model.fc2.bias.data = main_model.fc2.bias.data.clone()
            self.model.fc3.bias.data = main_model.fc3.bias.data.clone()

    def call_training(self, n_of_epoch):
        train_ds = TensorDataset(self.x_train, self.y_train)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        test_ds = TensorDataset(self.x_test, self.y_test)
        test_dl = DataLoader(test_ds, batch_size=self.batch_size * 2)

        for epoch in range(n_of_epoch):

            train_loss, train_accuracy = self.train(train_dl)
            test_loss, test_accuracy = self.validation(test_dl)

            logging.debug("Client: {}".format(self.id) + " | epoch: {:3.0f}".format(epoch+1) + " | train accuracy: {:7.5f}".format(train_accuracy) + " | test accuracy: {:7.5f}".format(test_accuracy))

    def train(self, train_dl):
        self.model.train()
        train_loss = 0.0
        correct = 0

        for data, target in train_dl:
            output = self.model(data)
            loss = self.criterion_info(output, target)
            self.optimizer_info.zero_grad()
            loss.backward()
            self.optimizer_info.step()

            train_loss += loss.item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

        return train_loss / len(train_dl), correct/len(train_dl.dataset)

    def validation(self, test_dl):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_dl:
                output = self.model(data)

                test_loss += self.criterion_info(output, target).item()
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(target.view_as(prediction)
                                         ).sum().item()

        test_loss /= len(test_dl)
        correct /= len(test_dl.dataset)

        return (test_loss, correct)

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            return self.model(data)

    def save_model(self, path):
        out_path = os.path.join(path, "model_{}".format(self.id))
        torch.save(self.model.state_dict(), out_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("conf_file",type=str)
    args = parser.parse_args()
    logging.debug("Running with configuration file {}".format(args.conf_file))

    conf_file = args.conf_file

    setup = Setup(conf_file)
    setup.run()
    setup.save()
