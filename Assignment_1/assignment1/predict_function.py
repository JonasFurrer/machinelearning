from sigmoid import sigmoid
import numpy as np


def predict_function(theta, X, y=None):
    """
    Compute predictions on X using the parameters theta. If y is provided
    computes and returns the accuracy of the classifier as well.

    """

    preds = None
    accuracy = None
    #######################################################################
    # TODO:                                                               #
    # Compute predictions on X using the parameters theta.                #
    # If y is provided compute the accuracy of the classifier as well.    #
    #                                                                     #
    #######################################################################
    
    preds = sigmoid(np.dot(X, theta))
    
    for i in range(np.size(preds)):
        if(preds[i] >= 0.5):
            preds[i] = 1
        else:
            preds[i] = 0
    
    if (y.all() != None):
        accuracy = 0
        for i in range(np.size(y)):
            if (y[i] == preds[i]):
                accuracy = (accuracy + 1)
        accuracy /= 1.0 * np.size(y)

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return preds, accuracy