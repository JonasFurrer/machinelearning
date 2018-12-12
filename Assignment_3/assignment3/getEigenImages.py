import numpy as np

def getEigenImages(images, basis):
    
    eigen_coefficients = None
    reconstruction = None
    
    #######################################################################
    # TODO:                                                               #
    #      Compute eigen coefficients and reconstruct the faces from      #
    #      coefficients                                                   #
    # Input:  images - images to compress                                 #
    #         basis - eigenbasis for compression                          #
    # Output: eigen_coefficients - coefficients corresponding to each     #
    #                              eigenvecvtor                           #
    #         reconstruction - compressed images                          #
    #######################################################################
  
    # Calculate eigen coefficients
    eigen_coefficients = np.dot(images, basis)
    
    #reconstruction 
    reconstruction = np.dot(eigen_coefficients, np.transpose(basis))
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return eigen_coefficients, reconstruction
