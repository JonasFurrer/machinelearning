from skimage.feature import hog
import numpy as np

def hog_features(X):
    """
    Extract HOG features from input images

    Args:
        X: Data matrix of shape [num_train, 577]

    Returns:
        hogs: Extracted hog features

    """
    
    hog_list = []
    
    for i in range(X.shape[0]):
        #######################################################################
        # TODO:                                                               #
        # Extract HOG features from each image and append them to the         #
        # hog_list                                                            #
        #                                                                     #
        # Hint: Make sure that you reshape the input features to size (24,24) #
        #       Make sure you add an intercept term to the extracted features #
        #                                                                     #
        #######################################################################
        
        image = np.reshape(X[i,:],(24,24))
        hog_list.append(hog(image))

        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################
        
    hogs = np.stack(hog_list,axis=0)
        
    return hogs