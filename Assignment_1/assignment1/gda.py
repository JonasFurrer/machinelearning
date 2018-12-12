from cost_function import cost_function
from math import pow
import numpy as np
import time


def gda(X, y):
    """
    Perform Gaussian Discriminant Analysis.

    Args:
        X: Data matrix of shape [num_train, num_features]
        y: Labels corresponding to X of size [num_train, 1]

    Returns:
        theta: The value of the parameters after logistic regression

    """

    theta = None
    phi = None
    mu_0 = None
    mu_1 = None
    sigma = None

    X = X[:, 1:]    # Note: We remove the bias term!
    start = time.time()

    #######################################################################
    # TODO:                                                               #
    # Perform GDA:                                                        #
    #   - Compute the values for phi, mu_0, mu_1 and sigma                #
    #                                                                     #
    #######################################################################

    indicator_1 = [[1 if y[i]==1 else 0] for i in range(X.shape[0])]
    indicator_0 = [[1 if y[i]==0 else 0] for i in range(X.shape[0])]

    # Scalar
    phi = 1.0/X.shape[0] * np.sum(indicator_1)
    
    # Creates 1xn array
    mu_0 = np.dot(np.transpose(indicator_0),X)/np.sum(indicator_0)
    
    # Creates 1xn array
    mu_1 = np.dot(np.transpose(indicator_1), X)/np.sum(indicator_1)
    
    mu = X - (np.outer(y,mu_1) + np.outer((1-y), mu_0))
    mu_transposed = np.transpose(mu)
    
    dot_product = np.dot(mu, mu_transposed)
    
    sigma = dot_product

    #mu = np.zeros(X.shape[1])
    #if(y[0] == 1):
        #mu = mu_1
    #if(y[0] == 0):
        #mu = mu_0
   
    #mu = [[np.concatenate((mu,mu_1), axis=0) if y[i]==1 else np.concatenate((mu,mu_0), axis=0)] for i in range(X.shape[0])] 
    #for i in range(1,X.shape[0]):
        #mu = np.concatenate((mu,mu_1), axis=0) if y[i]==1 else np.concatenate((mu,mu_0), axis=0)
        
    #sigma = (X-mu)@(X-mu)
    #sigma = sigma * 1.0/X.shape[0]
   

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    # Compute theta from the results of GDA
    sigma_inv = np.linalg.inv(sigma)
    quad_form = lambda A, x: np.dot(x.T, np.dot(A, x))
    b = 0.5*quad_form(sigma_inv, mu_0) - 0.5*quad_form(sigma_inv, mu_1) + np.log(phi/(1-phi))
    w = np.dot((mu_1-mu_0), sigma_inv)
    theta = np.concatenate([[b], w])
    exec_time = time.time() - start

    # Add the bias to X and compute the cost
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    loss = cost_function(theta, X, y)

    print('Iter 1/1: cost = {}  ({}s)'.format(loss, exec_time))

    return theta, None
