FROM tensorflow/tensorflow

ENV script=$attack_script
ENV setup=$setup_script
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
 wget https://github.com/fpinell/sec_federated_learning/archive/test.zip && \
 unzip test.zip && \
 rsync -a sec_federated_learning-test/ fedexp/ && \
 rm test.zip && \
 rm -r sec_federated_learning-test && \
 cd /home/fedexp/src && \
 python $script $setup
