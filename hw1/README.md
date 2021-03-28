# CSE 490/599 G1 Homework 1 #

Welcome fellow roboticists,

For this first assignment, we're going to be implementing part of a standard "multi-layer perceptron" aka Linear layers plus some non-linearities.
In this homework you will learn the ins and outs of how backprop actually works i.e. how gradients flow through the network.
You will train a classifier on MNIST, a common image dataset on hand-written numbers.
In the PyTorch part, you will visualize some of these numbers if you have never seen them before.
It's actually pretty impressive that we can make a neural net that reads handwriting.

We have provided some tests for checking your implementation. The tests are intentionally missing some cases, but feel free to write more tests yourself.
To run the tests from the outermost directory, simply run
```bash
pytest tests/hw1_tests
```
Or to run them for an individual file (for example test_linear_layer), run
```bash
pytest tests/hw1_tests/test_linear_layer.py
```

## Rules ##
1. You may not use PyTorch or any other deep learning package in parts 1-5 of the homework. Only Numpy and Numba are allowed. Functions like numpy.matmul are fine to use.
1. You may only modify the files we mention (those in the [submit.sh](submit.sh) script). We will not grade files outside of these.
1. You may not change the signatures or return types of `__init__`, `forward`, `backward`, or `step` in the various parts or you will fail our tests. You may add object fields (e.g. `self.data`) or helper functions within the files.
1. You may talk with others about the homework, but you must implement by yourself.
1. Feel free to test as you go and modify whatever you want, but we will only grade the files from [submit.sh](submit.sh).

## 1. Layers ##

If you check out [nn/layers/layer.py](nn/layers/layer.py) you will see a lot of complicated python-foo. You can pretty much ignore all of it.
Importantly though, all of your neural network based operations will inherit from `Layer`. This helps create and track the computation graph.
When you overload `Layer` you will need to implement a `forward` and `backward` function.
There is also a `parent` which helps create the graph. For this homework you can ignore that field. It has been taken care of by the `SequentialLayer`.

The `forward` function should take an input array and return a new array (not in place).

The `backward` function will take the partial gradients from the above layer, update the gradients of any parameters affected by the layer, and return the new gradients with respect to the input.
For this homework, each backward you implement should return a single array.

You can optionally also implement `initialize` and `selfstr`.

## 2. Parameters ##

These are special data holders for the weights (and biases) of the network. They will help you keep the forward weights and backward weight gradients straight.
Look at [nn/parameter.py](nn/parameter.py) to see what they hold. For forward passes, you will need to access the `param.data` field, and for backward, you will need `param.grad`.
Note that calling `param.grad = ...` actually does the `+=` operation in order to accumulate gradients (this will become more useful in later homeworks).

## 3. Writing your first Layers ##

### 3.1 Linear Layer ###

Open [nn/layers/linear_layer.py](nn/layers/linear_layer.py). Implementing the linear layer should be pretty straightforward.
Implement both the `forward` and `backward` function. You should not include a nonlinearity here (that will be somewhere else).
Also take a look at `selfstr`. This includes some extra information that will print when you call `print` on the layer or the network.
You don't need to change that, but you might want to do similar things in other layers.

To update the gradients of a parameter, just do `param.grad = newval`. No need to change the parameter itself here. That will be done in the optimization step.

You can expect the `LinearLayer` to take as input a 2D array of `(batch X features)` and it should return `(batch X output channels)`

### 3.2 ReLU Layer ###

ReLU is a pretty simple operation, but we will implement it in two different ways. DO NOT implement ReLU in place (you can use in-place operations, but you should not in-place modify the input data itself).
Since the gradient is undefined at 0 but is 0 for values less than 0, we will define the gradient at 0 to be 0 for simplicity.

ReLU Layers (and all the non-linearities you implement) should accept arrays of arbitrary shape.

Fill in the code for the ReLU function using standard Numpy operations. Your code should work on matrices of any shape.

## 4. Writing your first Loss Layer and SGD Updates ##
Congratulations, you now have two pieces which you can combine to make a fully functioning neural network. Now we need to make a way to update the weights.

### 4.1 Softmax Cross Entropy Loss Forward ###
Open [nn/layers/losses/softmax_cross_entropy_loss_layer.py](nn/layers/losses/softmax_cross_entropy_loss_layer.py).
Implement the forward pass. To avoid underflow/overflow, you should first subtract the max of each row.
Because we are using the Softmax function, we can prove that these two inputs should give equivalent results.
```math
\pi_i
= \frac{ \exp(x_i - b + b) }{ \sum_{j=1}^n \exp(x_j - b + b) }
= \frac{ \exp(x_i - b) \exp(b) }{ \sum_{j=1}^n \exp(x_j - b) \exp(b) }
= \frac{ \exp(x_i - b) }{ \sum_{j=1}^n \exp(x_j - b) }
```

By combining the Softmax and the Cross Entropy, we can actually implement a more stable loss as well. First we will implement Log Softmax (n is the size of the label dimension).
```math
\begin{aligned}
\log\left(\frac{e^{x_j}}{\sum_{i=1}^{n} e^{x_i}}\right) &= \log(e^{x_j}) - \log\left(\sum_{i=1}^{n} e^{x_i}\right) \\
&= x_j - \log\left(\sum_{i=1}^{n} e^{x_i}\right)
\end{aligned}
```

Finally, we can implement the Cross Entropy of the Softmax with the label.
```math
H(p,q) = -\sum_{i=1}^n p(i) log(q(x_i))
```
Where $`p(i)`$ is the label probability and $`log(q(x_i))`$ is the Log Softmax of the inputs. Since the probabilities are actually input as target integers, the probabilities will be a one-hot encoding of those targets.
Alternatively, you can use the target integers as indices from the Log Softmax array. Finally, be sure to implement both `mean` and `sum` reduction.

For the first homework, you can expect the input to be 2D (batch x class) and the label to be 1D (batch). However you will get bonus points if you correctly implement it for arbitrary dimensions (warning, harder than it sounds).
Hint: use numpy moveaxis and reshape to put the class dimension at the end and convert it to (batch x class), then undo after the computations.

### 4.2 Softmax Cross Entropy Loss Backward ###
Since the output of the forward function should be a float, the backward won't take any arguments. Instead, you should use some class variables to store relavent values from the forward pass in order to use them in the backward pass.
With some [fancy math](https://www.ics.uci.edu/~pjsadows/notes.pdf), we can show that the gradient of the loss wrt the logits is actually quite simple.
```math
\frac{\partial L}{\partial x_i} = q(x_i) - p(i)
```
Where $`p(i)`$ is the label probability and $`q(x_i)`$ is the Softmax of the inputs. Remember to scale the loss appropriately if the reduction was mean.


### 4.3 SGD Update ###
Open [nn/optimizers/sgd_optimizer.py](nn/optimizers/sgd_optimizer.py).
Recall that each `Parameter` has its own `data` and `grad` variables. Based on the other parts you wrote, the gradients should already be ready inside the `Parameter`. Now we just have to use them to update the weights.

Our normal SGD update with learning rate η is:

```math
w \Leftarrow w - \eta * \frac{\partial L}{\partial w}
```

With this done, we can now train our first neural network! Open [hw1/main.py](hw1/main.py). We have already provided code to train and test a simple three layer neural network.
Have a look at the pretty standard training loop. First, you get the data. Then you call the forward function on the network to get its outputs. Finally, you zero the previous gradients, call backward on the network, and update the weights.
You can run it by calling
```bash
cd hw1
python main.py
```
After 1 epoch, you should see about 70% test accuracy. After 10 epochs, you should see about 90% accuracy.

## Turn it in ##

First `cd` to the `hw1` directory. Then run the `submit.sh` script by running:

```bash
bash submit.sh
```

This will create the file `submit.tar.gz` in your directory with all the code you need to submit. The command will check to see that your files have changed relative to the version stored in the `git` repository. If it hasn't changed, figure out why, maybe you need to download your ipynb from google?

Submit `submit.tar.gz` in the file upload field for Homework 1 on Canvas.
