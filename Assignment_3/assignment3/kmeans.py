import numpy as np
import time


def kmeans(X, k, max_iter=100):
    """
    Perform k-means clusering on the data X with k number of clusters.

    Args:
        X: The data to be clustered of shape [n, num_features]
        k: The number of cluster centers to be used

    Returns:
        centers: A matrix of the computed cluster centers of shape [k, num_features]
        assign: A vector of cluster assignments for each example in X of shape [n] 

    """

    centers = None
    assign = None
    i=0
    
    start = time.time()


    #######################################################################
    # TODO:                                                               #
    # Perfom k-means clustering of the input data X and store the         #
    # resulting cluster-centers as well as cluster assignments.           #
    #                                                                     #
    #######################################################################
    
    # Here we choose k training examples randomly and set the cluster centers
    # equal to the values of these examples to initialize
    centers = np.zeros((k,X.shape[1]))
    rand = np.random.randint(X.shape[0], size=k)
    
    for i in range(k):
        centers[i,:] = X[rand[i],:]
      
    # Calculate the distances between datapoints and centers
    # Broadcasting is used to efficiently calculate
    dist = np.array((k, X.shape[0]))
    dist = np.sqrt(((X - centers[:,np.newaxis])**2).sum(axis=2))
    
    # Finally do the assignment of the datapoints to the nearest neighbour
    # argmin chooses the lowest value for every training example
    assign = np.argmin(dist, axis=0)
   
    # Just to test the broadcasting
    # print X
    # print centers
    # print X-centers[:,np.newaxis]
    
    # Now the centers are updated
    centers =  np.array([X[assign==i].mean(axis=0) for i in range(centers.shape[0])])

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    
    exec_time = time.time()-start
    print('Number of iterations: {}, Execution time: {}s'.format(i+1, exec_time))
    
    return centers, assign