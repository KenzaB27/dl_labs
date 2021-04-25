# DD2424 Deep Learning in Data Science course
Author: *Kenza Bouzid*

Solutions for labs for DL course at KTH. Each lab contains implementation of neural networks algorithms as well as notebooks with experiments.

## Lab1 - One Layer Network to classify CIFAR-10

### Mandatory 
- Back Propagation for 1 layer network
- Mini Batch gradient descent to optimize Cross Entropy Loss + L2 Regularization 

### Bonus

- Mini Batch gradient descent to optimize Hinge Loss + L2 Regularization
- Tricks and Avenues to improve the network's performance:
  - Early stopping 
  - Learning Rate Decay
  - Xavier Initialization
  - Data Shuffling and Reordering at the beginning of each epoch 


## Lab2 - 2 Layer Network with Cyclic Learning

### Mandatory 

- Back Propagation for 2 layer network
- Mini Batch gradient descent to optimize Cross Entropy Loss + L2 Regularization with cyclic learning

### Bonus

- Find the learning rate boundaries with LR range Test
- Tricks and Avenues to improve the network's performance:
  - Data Augmentation
  - Ensemble Learning: Majority voting 
  - Augmenting the hidden dimension

## Lab3 - k- Layer Network and Batch Normalization

### Mandatory

- Implement Back Propagation for a k-layer network with cyclic learning
- Enhance it with batch normalization
-  Coarse-to-fine Search for Regularization term

#### Bonus 

* Tricks and Avenues to improve the network's performance:
  - Data Augmentation
  - Search for the best network architecture
  - Dropout

## Lab4 - RNN
Train a vanilla RNN with outputs, as described using the text from the book The Goblet of Fire by J.K. Rowling.
The variation of SGD you will use for the optimization will be AdaGrad.
* Major Components: 
  - Preparing the data: one hot encoding 
  - Back Propagation for vanilla RNN 
  - AdaGrad
  - synthesizing text from RNN