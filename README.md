# tensor-test
This repository is meant for testing/learning TensorFlow

1) <b>Linear-model:</b>
    - Linear Model with Gradient Descent optimizer

    - To run (with time taken to run script):
        time python3 linear-model.py

2) <b>Contrib-learn:</b>
    - Basically the same as linear-model.py, but using contrib-learn library.

    - To run (with time taken to run script):
        time python3 contrib-learn.py

3) <b>MNIST-NN:</b>
    - Neural Network trained on the MNST dataset. Using Relu activation function and default learning rate 0.0001

    - To run (with time taken to run script):
        time python3 mnist-nn.py

    - To view TensorBoard run:
         tensorboard --logdir=./logs/nn_logs
