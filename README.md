# sec_federated_learning

we prove that federated learning systems can be turned into covert channels to implement a stealth communication infrastructure.
The main intuition is that an attacker can injects a bias in the global model by submitting purposely crafted samples.
The effect of the model bias is negligible to the other participants.
Yet, the bias can be measured and used to transmit a single bit through the model.


## Run with docker

```
docker run -it -v $(pwd):/home/fedexp gabrielec/fedexp /bin/bash
```
