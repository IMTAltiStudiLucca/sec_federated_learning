FROM tensorflow/tensorflow

RUN mkdir /home/fedexp
RUN python -m pip install --upgrade pip
RUN pip install pandas
RUN pip install sklearn
RUN pip install scikit-image
RUN pip install matplotlib
RUN pip install keras
RUN pip install torch
RUN pip install torchvision
CMD cd /home
CMD wget https://github.com/fpinell/sec_federated_learning/archive/main.zip
CMD unzip main.zip
CMD mv sec_federated_learning-main/* fedexp
CMD rm -r sec_federated_learning-main
WORKDIR /home/fedexp/src
