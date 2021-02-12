# Tests Strategy

*Jan 30, 2021*

### Assumptions:

- We assume to have a binary symmetric channel (BSC). This implies that we have the same error probability for 0s and 
  1s. 
- **Score-attack**:
  
  - we assume that the sender, and the receiver know a priori which is the digit that they should use to communicate 
    over the covert channel.

- **Label-attack**:

  - we assume that the sender, and the receiver know a priori which couple of digit within the MNIST dataset are 
    selected for the covert channel.
      

### Parameters:
  These parameters are selected to define different test scenarios.
  - **p**  = client probability of selection
    - p = 0.1, 0.2, 0.3, 0.4, 0.5
  - **K**  = number of bit transmitted
    - K = 10 (for figures), 1000, 10000
  - <s>**Rc** = number of recalibration slot
    - Rc = 1, K/100, K/1000 </s>
  - **TN** = type of network
    - TN = NN or CNN

### Outputs:
  we save these output at the end of each test.
    
- **BER** = bit error rate = transmission errors / K
- **C**   = covert channel capacity.

### Figures:

- **Score-attack**:

  - Figure 1: Score vs Time (epochs)
  - Figure 2: BER vs Channel capacity by varying <s>Rc</s> TN
  
- **Label-attack**:

  - Figure 1: Label vs Time (epochs)
  - Figure 2: BER vs Channel capacity by varying <s>Rc</s> TN