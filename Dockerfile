FROM tensorflow/tensorflow

ARG attack_script
ENV script=$attack_script
RUN mkdir /home/fedexp
RUN python -m pip install --upgrade pip
RUN pip install pandas
RUN pip install sklearn
RUN pip install scikit-image
RUN pip install matplotlib
RUN pip install keras
RUN pip install torch
RUN pip install torchvision
RUN apt-get update && apt-get install -y wget unzip rsync
ENTRYPOINT cd /home && \
 wget https://github.com/fpinell/sec_federated_learning/archive/main.zip && \
 unzip main.zip && \
 rsync -a sec_federated_learning-main/ fedexp/ && \
 rm main.zip && \
 rm -r sec_federated_learning-main && \
 cd /home/fedexp/src && \
 python $script attack_setup.yaml
