# pytorch/pytorch (3GB!) should be the right one (there also is a CUDA version) but it is too large
FROM python

COPY ./data/mnist /home/fedexp/mnist
COPY ./src/all.py /home/fedexp
RUN pip install numpy
RUN pip install pandas
RUN pip install sklearn
RUN pip install torch
RUN pip install tensorboard matplotlib pathlib requests torchvision
CMD cd /home/fedexp/ ; python all.py
