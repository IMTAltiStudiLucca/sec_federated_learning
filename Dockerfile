# pytorch/pytorch (3GB!) should be the right one (there also is a CUDA version) but it is too large
FROM tensorflow/tensorflow

COPY . /home/fedexp
RUN python -m pip install --upgrade pip
RUN pip install pandas
RUN pip install sklearn
RUN pip install matplotlib
RUN pip install keras
RUN pip install torch
RUN pip install torchvision
WORKDIR /home/fedexp
