import numpy as np
import pandas as pd
from copy import deepcopy as dcopy
import sys
import os

np.random.seed(42)


def softmax(logits):
    """
    Implement the softmax function
    Inputs:
    - logits : A numpy-array of shape (n * number_of_classes )
    Returns:
    - probs : Normalized probabilities for each class,A numpy array of shape (n * number_of_classes)
    """
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1).reshape(-1, 1)


def cross_entropy_loss(probs, target):
    """
    Implement the cross Entropy loss function

    Inputs:
    - probs : A numpy-array of shape ( n * number_of_classes )
    - target : A numpy-array of shape ( n, )
    Returns:
    - loss : A scalar describing the mean cross-entropy loss over the batch
    """
    n = probs.shape[0]
    oh_target = np.eye(probs.shape[1])[target]
    return - np.sum(np.log(probs + 1e-10) * oh_target) / n


def regularization_loss(weights, biases):
    """
    Inputs:
    - weights : weights of the network
    - biases : biases of the network

    Returns : the regularization loss
    """
    loss = 0
    for w in weights:
        loss += np.sum(w * w)
    for b in biases:
        loss += np.sum(b * b)
    return loss


def loss_fn(probs, target, weights, biases, _lambda):
    """
    function to calculate total loss
    Inputs:
    - probs : output of the neural network , a numpy array of shape (n, number_of_classes)
    - target : ground truth , a numpy array of shape (n,)
    - weights : weights of the network , an array containing weight matrices for all layers.(shape of the weight matrices vary according to the layer)
    - biases : biases of the network
    - _lambda : regularization constant
    Returns:
    - returns the total loss i.e - Mean cross-entropy loss + _lambda*regularization loss

    Note : This function is not being used anywhere.This will be used just for grading
    """
    return cross_entropy_loss(probs, target) + _lambda * regularization_loss(weights, biases)


def check_accuracy(prediction, target):
    """
    Find the accuracy of the prediction
    Inputs:
    - prediction : most-probable class for each datapoint, a numpy array of dimension (n, )
    - target : ground truth , a numpy array of dimension (n,)

    Returns :

    - accuracy : a scalar between 0 and 1 ,describing the accuracy , where 1 means prediction is same as ground truth

    """
    return (prediction == target).mean()


class Neural_Net():

    def __init__(self, num_layers, num_units, input_dim, output_dim):
        '''
        Initialize the weights and biases of the network
        Inputs:
        - num_layers : Number of HIDDEN layers
        - num_units : Number of units in each hidden layer
        - input_dim : Number of features i.e your one batch is of shape (batch_size * input_dim)
        - output_dim : Number of units in output layer , i.e number of classes
        '''
        self.weights = []
        self.biases = []
        self.weights.append(np.random.uniform(-1, 1, (input_dim, num_units)))
        self.biases.append(np.random.uniform(-1, 1, (num_units)))

        for _ in range(num_layers - 1):
            self.weights.append(
                np.random.uniform(-1, 1, (num_units, num_units)))
            self.biases.append(np.random.uniform(-1, 1, (num_units)))

        self.weights.append(np.random.uniform(-1, 1, (num_units, output_dim)))
        self.biases.append(np.random.uniform(-1, 1, (output_dim)))

        self.activations = []

    def relu(self, x):
        return np.maximum(x, 0)

    def forward(self, X):
        """
        Perform the forward step of backpropagation algorithm
        Inputs :
        - X : a numpy array of dimension (n , number_of_features)
        Returns :
        - probs : the predictions for the data.For each training example, probs 
                 contains the probability distribution over all classes.
                 a numpy array of dimension (n , number_of-classes)

        Note : you might want to save the activation of each layer , which will be required during backward step

        """
        self.activations = []
        x = dcopy(X)
        for i, weight, bias in zip(range(len(self.weights)),
                                   self.weights, self.biases):
            x = x@weight + bias
            if i == len(self.weights) - 1:
                x = softmax(x)
            else:
                x = self.relu(x)
            self.activations.append(dcopy(x))

        return x

    def backward(self, X, probs, targets, _lambda):
        """
        perform the backward step of backpropagation algorithm and calculate the gradient of loss function with respect to weights and biases (dL/dW,dL/db)
        Inputs:
        - X : a single batch, a numpy array of dimension (n , number_of_features)
        - probs : predictions for a single batch , a numpy array of dimension ( n, num_of_classes)
        - targets : ground truth , a numpy array of dimension having dimension ( n, )
        - _lambda : regularization constant

        Returns:

        - dW - gradient of total loss with respect to weights, 

        - db - gradient of total loss with respect to biases, 

        Note : Ideally , you would want to call the forward function for the same batch or data before calling the backward function,So that
               the accumulated activations are consistent and not stale.

               Also Don't forget to take regularization into account while calculating gradients

        """
        # self.forward(X)

        dW = []
        dB = []
        celoss = cross_entropy_loss(probs, targets)
        oh_targets = np.eye(probs.shape[1])[targets]

        dZi = probs - oh_targets

        for i in reversed(range(len(self.weights))):
            if i >= 1:
                dWi = self.activations[i - 1].T @ dZi + _lambda * self.weights[i]
                dBi = dZi.sum(axis=0).reshape(-1) + _lambda * self.biases[i]
                dAi = (dZi @ self.weights[i].T)
                dZi = dcopy(dAi)
                dZi[self.activations[i - 1] < 0] = 0
            else:
                dWi = X.T @ dZi / len(X)
                dBi = dZi.sum(axis=0).reshape(-1) / len(X)
            dW.append(dcopy(dWi / len(X)))
            dB.append(dcopy(dBi / len(X)))

        dW = dW[::-1]
        dB = dB[::-1]

        return dW, dB

    def train(self, optimizer, _lambda, batch_size, max_epochs, train_input, train_target, val_input, val_target, patience):
        """
        Here you will run backpropagation for max_epochs number of epochs and evaluate 
        the neural network on validation data.For each batch of data in each epoch you 
        will do the forward step ,then backward step of backpropagation.And then you
        will update the gradients accordingly.

        Note : Most of the things here are already implemented.However, you are welcome to change it for part 2 of the assignment.
        """

        val_losses = []

        for epoch in range(max_epochs):
            idxs = np.arange(train_input.shape[0])
            np.random.shuffle(idxs)  # shuffle the indices
            # split into a number of batches
            batch_idxs = np.array_split(
                idxs, np.ceil(train_input.shape[0] / batch_size))

            for i in range(len(batch_idxs)):
                train_batch_input = train_input[
                    batch_idxs[i], :]  # input for a single batch

                train_batch_target = train_target[
                    batch_idxs[i]]  # target for a single batch

                # perform the forward step
                probs = self.forward(train_batch_input)

                # perform the backward step and calculate the gradients
                dW, db = self.backward(
                    train_batch_input, probs, train_batch_target, _lambda)

                self.weights, self.biases = optimizer.step(
                    self.weights, self.biases, dW, db)  # update the weights

            val_probs = self.forward(val_input)
            val_loss = cross_entropy_loss(val_probs, val_target)
            val_losses.append(val_loss)
            if val_loss == min(val_losses):
                best_model = dcopy(self)
                min_val_loss = val_loss
            if len(val_losses) >= patience:
                if min(val_losses[-patience:]) > min_val_loss:
                    # val losses have been going up
                    break

            if epoch % 5 == 0:
                train_probs = self.forward(train_input)
                val_probs = self.forward(val_input)
                train_loss = cross_entropy_loss(train_probs, train_target)
                val_loss = cross_entropy_loss(val_probs, val_target)
                train_acc = check_accuracy(
                    self.predict(train_input), train_target)
                val_acc = check_accuracy(self.predict(val_input), val_target)
                print("train_loss = {:.3f}, val_loss = {:.3f}, train_acc={:.3f}, val_acc={:.3f}".format(
                    train_loss, val_loss, train_acc, val_acc))
        train_probs = best_model.forward(train_input)
        val_probs = best_model.forward(val_input)
        train_loss = cross_entropy_loss(train_probs, train_target)
        val_loss = cross_entropy_loss(val_probs, val_target)
        return train_loss, val_loss, best_model

    def predict(self, X):
        """
        Predict the most probable classes for each datapoint in X
        Inputs : 
        - X : a numpy array of shape (n,number_of_features)
        Returns :
        - preds : Most probable class for each datapoint in X , a numpy array of shape (n,1)

        """
        return np.argmax(self.forward(X), axis=1)


class Optimizer(object):

    def __init__(self, learning_rate):
        """
        Initialize the learning rate
        """
        self.lr = learning_rate

    def step(self, weights, biases, delta_weights, delta_biases):
        """
        update the gradients
        Inputs :
        - weights : weights of the network
        - biases : biases of the network
        - delta_weights : gradients with respect to weights
        - delta_biases : gradients with respect to biases
        Returns :
        Updated weights and biases
        """
        for i in range(len(weights)):
            weights[i] -= self.lr * delta_weights[i]
            biases[i] -= self.lr * delta_biases[i]

        return weights, biases


def normalize(df):
    global phase, means, ranges
    df = np.array(df, "float64")
    if phase == "train":
        means = np.mean(df, axis=0)
        ranges = np.ptp(df, axis=0)
        zeros = np.where(ranges == 0)
        ranges[zeros] = 1
        means[zeros] = 0
        return (df - means) / ranges
    return (df - means) / ranges


def read_data():
    global phase

    df_train = pd.read_csv("data/train.csv")
    df_val = pd.read_csv("data/val.csv")
    df_test = pd.read_csv("data/test.csv")

    phase = "train"

    x_train = normalize(df_train.drop("letter", axis=1))
    y_train = df_train["letter"].apply(
        lambda x: ord(x) - ord('A'))

    phase = "val"

    x_val = normalize(df_val.drop("letter", axis=1))
    y_val = df_val["letter"].apply(
        lambda x: ord(x) - ord('A')).to_numpy()

    phase = "test"
    x_test = normalize(df_test)

    return x_train, y_train, x_val, y_val, x_test


if __name__ == '__main__':
    max_epochs = 10**4
    batch_size = 128
    learning_rate = 0.1
    num_layers = 2
    num_units = 512
    _lambda = 1e-5
    patience = 20

    train_x, train_y, val_x, val_y, test_x = read_data()
    net = Neural_Net(num_layers, num_units, train_x.shape[1], 26)
    optimizer = Optimizer(learning_rate=learning_rate)
    train_loss, val_loss, best_net = net.train(optimizer, _lambda, batch_size,
                                               max_epochs, train_x, train_y, val_x, val_y, patience)

    print(train_loss, val_loss)

    y_test_pred = best_net.predict(test_x).reshape(-1, 1)
    id_col = np.array(list(range(0, len(y_test_pred)))).reshape(-1, 1)
    np_test = np.concatenate((id_col, y_test_pred), axis=1)
    df_test = pd.DataFrame(np_test, index=None, columns=["id", "letters"])
    df_test["letters"] = df_test["letters"].apply(lambda x: chr(ord('A') + x))

    df_test.to_csv("test_pred.csv", index=None)
