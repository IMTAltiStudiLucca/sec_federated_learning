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

class Net2nn(nn.Module):
    def __init__(self):
        super(Net2nn, self).__init__()
        self.fc1=nn.Linear(784,200)
        self.fc2=nn.Linear(200,200)
        self.fc3=nn.Linear(200,10)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

class Setup:
  '''Read the dataset, instantiate the clients and the server
     It receive in input the path to the data, and the number of clients
  '''
  def __init__(self,data_path,n_clients=5,learning_rate=0.01,num_of_epochs=10,batch_size=32,momentum=0.9,random_clients=5):
    self.data_path = data_path
    self.n_clients = n_clients
    self.learning_rate = learning_rate
    self.num_of_epochs = num_of_epochs
    self.batch_size = batch_size
    self.momentum = momentum
    self.random_clients = random_clients

    self.train_amount=4500
    self.valid_amount=900
    self.test_amount=900
    self.list_of_clients = []

    self.X_train, self.y_train, self.X_test, self.y_test = self.__load_dataset()

    self.create_clients()

    self.server = Server(self.list_of_clients,self.random_clients,self.learning_rate,self.num_of_epochs,self.batch_size,self.momentum)


  def run(self,federated_runs=10):
    for i in range(federated_runs):
      print("{}th run of the federated learning".format(i))
      self.server.training_clients()
      self.server.update_averaged_weights()
      self.server.send_weights()

  def __load_dataset(self):
      (X_train, y_train), (X_test, y_test) = mnist.load_data()
      X_train = X_train.reshape(X_train.shape[0], 784)
      X_test = X_test.reshape(X_test.shape[0], 784)
      X_train = X_train.astype('float32')
      X_test  = X_test.astype('float32')
      X_train /= 255 # Original data is uint8 (0-255). Scale it to range [0,1].
      X_test  /= 255

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
      c = Client(i, X_trains[i], y_trains[i], X_tests[i], y_tests[i], self.learning_rate, self.momentum, self.batch_size)
      self.list_of_clients.append(c)

  def save_models(self):
  	self.server.save_model()
  	for c in self.list_of_clients:
  		c.save_model()



class Server:
  '''
  The Server class owns the central model.
  - It initializes the main model and it updates the weights to the clients
  - It handles the training among the clients
  - It receives the weights from clients and it averages them for its main
    model updating it
  '''
  def __init__(self,list_of_clients,random_clients,learning_rate=0.01,num_of_epochs=10,batch_size=32,momentum=0.9):
    self.list_of_clients = list_of_clients
    self.random_clients = random_clients
    self.learning_rate = learning_rate
    self.num_of_epochs = num_of_epochs
    self.batch_size = batch_size
    self.momentum = momentum

    self.selected_clients = []
    self.main_model = Net2nn()
    self.main_optimizer = torch.optim.SGD(self.main_model.parameters(), lr=self.learning_rate, momentum=self.momentum)
    self.main_criterion = nn.CrossEntropyLoss()
    self.send_weights()

  def send_weights(self):
    for c in self.list_of_clients:
      c.update_model_weights(self.main_model)

  def training_clients(self):
    self.selected_clients = random.sample(self.list_of_clients,math.floor(len(self.list_of_clients)*self.random_clients))
    for c in self.selected_clients:
      c.call_training(self.num_of_epochs)

  def get_averaged_weights(self):

    fc1_mean_weight = torch.zeros(size=self.list_of_clients[0].model.fc1.weight.shape)
    fc1_mean_bias = torch.zeros(size=self.list_of_clients[0].model.fc1.bias.shape)

    fc2_mean_weight = torch.zeros(size=self.list_of_clients[0].model.fc2.weight.shape)
    fc2_mean_bias = torch.zeros(size=self.list_of_clients[0].model.fc2.bias.shape)

    fc3_mean_weight = torch.zeros(size=self.list_of_clients[0].model.fc3.weight.shape)
    fc3_mean_bias = torch.zeros(size=self.list_of_clients[0].model.fc3.bias.shape)

    with torch.no_grad():
        for c in self.selected_clients:
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


  def predict(self,data):
    self.main_model.eval()
    with torch.no_grad():
      return self.main_model(data)


  def save_model(self):
  	path = "../models/main_model"
  	torch.save(self.main_model.state_dict(), path)

class Client:
  '''A client who has its own dataset to use for training.
     The main methods of the Client class are:
     - Load the data
     - Get the weights from the server
     - Train the model
     - Return the weights to the server
     - Get a sample and perform a prediction with the probabilities for each class
      '''
  def __init__(self,id,x_train,y_train,x_test, y_test,learning_rate,momentum,batch_size=32,weights=None):
    self.id = id
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.batch_size = batch_size
    self.weights = weights
    # training and test can be splitted inside the client class
    # now we are passing them while instantiate the class

    x_train, y_train, x_test, y_test = map(torch.tensor, (x_train, y_train, x_test, y_test))
    y_train = y_train.type(torch.LongTensor)
    y_test = y_test.type(torch.LongTensor)

    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test

    self.model_name="model"+str(self.id)
    self.model=Net2nn()

    self.optimizer_name="optimizer"+str(self.id)
    self.optimizer_info = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

    self.criterion_name = "criterion"+str(self.id)
    self.criterion_info = nn.CrossEntropyLoss()

  def update_model_weights(self,main_model):
    with torch.no_grad():
      self.model.fc1.weight.data = main_model.fc1.weight.data.clone()
      self.model.fc2.weight.data = main_model.fc2.weight.data.clone()
      self.model.fc3.weight.data = main_model.fc3.weight.data.clone()

      self.model.fc1.bias.data = main_model.fc1.bias.data.clone()
      self.model.fc2.bias.data = main_model.fc2.bias.data.clone()
      self.model.fc3.bias.data = main_model.fc3.bias.data.clone()

  def call_training(self,n_of_epoch):
    train_ds = TensorDataset(self.x_train, self.y_train)
    train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

    test_ds = TensorDataset(self.x_test, self.y_test)
    test_dl = DataLoader(test_ds, batch_size= self.batch_size * 2)

    for epoch in range(n_of_epoch):

        train_loss, train_accuracy = self.train(train_dl)
        test_loss, test_accuracy = self.validation(test_dl)

        print("Client {}".format(self.id) + " | epoch: {:3.0f}".format(epoch+1) + " | train accuracy: {:7.5f}".format(train_accuracy) + " | test accuracy: {:7.5f}".format(test_accuracy))

  def train(self,train_dl):
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

  def validation(self,test_dl):
    self.model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_dl:
            output = self.model(data)

            test_loss += self.criterion_info(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(test_dl)
    correct /= len(test_dl.dataset)

    return (test_loss, correct)

  def predict(self,data):
    self.model.eval()
    with torch.no_grad():
      return self.model(data)

  def save_model(self):
  	path = "../models/model_client_{}".format(self.id)
  	torch.save(self.model.state_dict(), path)


if __name__ == '__main__':

	data_path = ""
	n_clients=10
	learning_rate=0.01
	num_of_epochs=10
	batch_size=32
	momentum=0.9
	random_clients=0.5

	n_of_federated_runs = 1

	setup = Setup(data_path,n_clients,learning_rate,num_of_epochs,batch_size,momentum,random_clients)
	setup.run(n_of_federated_runs)
	setup.save_models()


# init con modelli già allenati
