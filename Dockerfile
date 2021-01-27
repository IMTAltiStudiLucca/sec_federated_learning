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
RUN apt-get update && apt-get install -y wget unzip
ENTRYPOINT cd /home && \
 wget https://github.com/fpinell/sec_federated_learning/archive/main.zip && \
 unzip main.zip && \
 mv sec_federated_learning-main/* fedexp && \
 rm main.zip && \
 rm -r sec_federated_learning-main && \
 cd /home/fedexp/src && \
 reset && \
 python label-attack.py attack_setup.yaml
