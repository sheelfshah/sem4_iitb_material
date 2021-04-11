import numpy as np
import pandas as pd

np.random.seed(42)


def softmax(logits):
    """
    Implement the softmax function
    Inputs:
    - logits : A numpy-array of shape (n * number_of_classes )
    Returns:
    - probs : Normalized probabilities for each class,A numpy array of shape (n * number_of_classes)
    """
    pass



def cross_entropy_loss(probs,target):
    """
    Implement the cross Entropy loss function
    
    Inputs:
    - probs : A numpy-array of shape ( n * number_of_classes )
    - target : A numpy-array of shape ( n, )
    Returns:
    - loss : A scalar describing the mean cross-entropy loss over the batch
    """
    pass
    


def regularization_loss(weights,biases):
    """
    Inputs:
    - weights : weights of the network
    - biases : biases of the network
    
    Returns : the regularization loss
    """
    pass



def loss_fn(probs,target,weights,biases,_lambda):
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
    pass
    


def check_accuracy(prediction,target):
    """
    Find the accuracy of the prediction
    Inputs:
    - prediction : most-probable class for each datapoint, a numpy array of dimension (n, )
    - target : ground truth , anumpy array of dimension (n,)
    
    Returns :
    
    - accracy : a scalar between 0 and 1 ,describing the accuracy , where 1 means prediction is same as ground truth
    
    """



class Neural_Net():
    def __init__(self,num_layers,num_units,input_dim,output_dim):
        '''
        Initialize the weights and biases of the network
        Inputs:
        - num_layers : Number of HIDDEN layers
        - num_units : Number of units in each hidden layer
        - input_dim : Number of features i.e your one batch is of shape (batch_size * input_dim)
        - output_dim : Number of units in output layer , i.e number of classes
        '''
        pass
        
        
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
        
        pass
    
    def backward(self, X, probs,targets, _lambda):
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
        pass
    
    
    def train(self, optimizer, _lambda, batch_size, max_epochs,train_input, train_target,val_input, val_target):
        """
        Here you will run backpropagation for max_epochs number of epochs and evaluate 
        the neural network on validation data.For each batch of data in each epoch you 
        will do the forward step ,then backward step of backpropagation.And then you
        will update the gradients accordingly.

        Note : Most of the things here are already implemented.However, you are welcome to change it for part 2 of the assignment.
        """
        for epoch in range(max_epochs):
            idxs = np.arange(train_input.shape[0])
            np.random.shuffle(idxs) #shuffle the indices
            batch_idxs = np.array_split(idxs,np.ceil(train_x.shape[0]/batch_size)) #split into a number of batches

            for i in range(len(batch_idxs)):
                train_batch_input = train_input[batch_idxs[i],:] # input for a single batch

                train_batch_target = train_target[batch_idxs[i]] # target for a single batch

                probs = self.forward(train_batch_input) # perform the forward step

                dW,db = self.backward(train_batch_input,probs, train_batch_target,_lambda) #perform the backward step and calculate the gradients

                self.weights,self.biases = optimizer.step(self.weights,self.biases,dW,db) # update the weights
            if epoch % 5 == 0 :
                train_probs = self.forward(train_input)
                val_probs = self.forward(val_input)
                train_loss = cross_entropy_loss(train_probs,train_target)
                val_loss = cross_entropy_loss(val_probs,val_target)
                train_acc = check_accuracy(self.predict(train_input),train_target)
                val_acc = check_accuracy(self.predict(val_input),val_target)
                print("train_loss = {:.3f}, val_loss = {:.3f}, train_acc={:.3f}, val_acc={:.3f}".format(train_loss,val_loss,train_acc,val_acc))

                    
    def predict(self,X):
        """
        Predict the most probable classes for each datapoint in X
        Inputs : 
        - X : a numpy array of shape (n,number_of_features)
        Returns :
        - preds : Most probable class for each datapoint in X , a numpy array of shape (n,1)
        
        """
        pass


class Optimizer(object):

    def __init__(self, learning_rate):
        """
        Initialize the learning rate
        """
        pass

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
        pass



def read_data():
    """
    Read the train,validation and test data
    """
    pass



if __name__ == '__main__':
    max_epochs = 100
    batch_size = 128
    learning_rate = 0.1
    num_layers = 1
    num_units = 64
    _lambda = 1e-5
    
    train_x,train_y,val_x,val_y,test_x = read_data()
    net = Neural_Net(num_layers,num_units,train_x.shape[1],26)
    optimizer = Optimizer(learning_rate=learning_rate)
    net.train(optimizer,_lambda,batch_size,max_epochs,train_x,train_y,val_x,val_y)
    
    test_preds = net.predict(test_x)
    

