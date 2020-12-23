from federated_learning import Client
from bitarray import bitarray

class Sender(Client):


    def __init__(self,x_train,y_train,x_test, y_test,learning_rate,momentum,batch_size=32,weights=None):
        self.msg = bitarray(8)
        super().__init__(self,"Sender",x_train,y_train,x_test, y_test,learning_rate,momentum,batch_size,weights)


class Receiver(Client):

    def __init__(self,x_train,y_train,x_test, y_test,learning_rate,momentum,batch_size=32,weights=None):
        super().__init__(self,"Receiver",x_train,y_train,x_test, y_test,learning_rate,momentum,batch_size,weights)





def run():
    # 1. parse arguments

    # 2. create Setup

    # 3. run N rounds OR load pre-trained models

    # 4. create Sender and Receiver

    # 5. fuzz for channel baselines

    # 6. perform channel calibration

    # 7. start transmitting

if __name__ == '__main__':
    run()
