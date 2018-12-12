import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import time


def em_mog(X, k, max_iter=20):
    """
 
    Args:
        X: The data used for training [n, num_features]
        k: The number of gaussians to be used

    Returns:
        phi: A vector of probabilities for the latent vars z of shape [k]
        mu: A marix of mean vectors of shape [k, num_features] 
        sigma: A list of length k of covariance matrices each of shape [num_features, num_features] 
        w: A vector of weights for the k gaussians per example of shape [n, k] (result of the E-step)
        
    """

    # Initialize variables
    mu = None
    sigma = [np.eye(X.shape[1]) for i in range(k)]
    phi = np.ones([k,])/k
    ll_prev = float('inf')
    start = time.time()
    
    #######################################################################
    # TODO:                                                               #
    # Initialize the means of the gaussians. You can use K-means!         #
    #######################################################################
    
    kmeans = KMeans(n_clusters=k, max_iter=100, random_state=0).fit(X)
    mu = kmeans.cluster_centers_
    
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    for l in range(max_iter): 
        # E-Step: compute the probabilities p(z==j|x; mu, sigma, phi)
        w = e_step(X, mu, sigma, phi)
        
        # M-step: Update the parameters mu, sigma and phi
        phi, mu, sigma = m_step(w, X, mu, sigma, phi, k) 
        
        # Check convergence
        ll = log_likelihood(X, mu, sigma, phi)
        print('Iter: {}/{}, LL: {}'.format(l+1, max_iter, ll))
        if ll/ll_prev > 0.999:
            print('EM has converged...')
            break
        ll_prev = ll
    
    # Get stats
    exec_time = time.time()-start
    print('Number of iterations: {}, Execution time: {}s'.format(l+1, exec_time))
    
    # Compute final assignment
    w = e_step(X, mu, sigma, phi)
    
    return phi, mu, sigma, w



def log_likelihood(X, mu, sigma, phi):
    """
    Returns the log-likelihood of the data under the current parameters of the MoG model.
    
    """
    ll = None
    
    #######################################################################
    # TODO:                                                               #
    # Compute the log-likelihood of the data under the current model.     #
    # This is used to check for convergnence of the algorithm.            #
    #######################################################################
    
    res = 0
    
    for i in range(4): 
        # Calculate mean vector for a single Gaussian
        mu_single = mu[i]
        
        # Calculate cov matrix for a single Gaussian
        sigma_single = sigma[i]
        
        # Calculate phi for a single Gaussian
        phi_single = phi[i]
        
        # Calculate probability density function * phi
        pdf = multivariate_normal.pdf(X, mean=mu_single, cov=sigma_single)
        res += np.sum(np.log(np.dot(pdf,phi_single)))
        
    ll = res
    
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    
    return ll
                    
    
def e_step(X, mu, sigma, phi):
    """
    Computes the E-step of the EM algorithm.

    Returns:
        w:  A vector of probabilities p(z==j|x; mu, sigma, phi) for the k 
            gaussians per example of shape [n, k] 
    """
    w = None
    
    #######################################################################
    # TODO:                                                               #
    # Perform the E-step of the EM algorithm.                             #
    # Use scipy.stats.multivariate_normal.pdf(...) to compute the pdf of  #
    # of a gaussian with the current parameters.                          # 
    #######################################################################
   
    # Convert form list to array to better work with it
    sigma = np.array(sigma)
    
    # Define variable for the temp. denominator
    denom = np.zeros((X.shape[0], ))
    
    # Define variable for temp. numberator
    weights = np.zeros((X.shape[0], 4))
    
    for i in range(4): 
        # Calculate mean vector for a single Gaussian
        mu_single = mu[i]
        
        # Calculate cov matrix for a single Gaussian
        sigma_single = sigma[i]
        
        # Calculate phi for a single Gaussian
        phi_single = phi[i]
        
        # Calculate numerator (probability density function * phi)
        pdf = multivariate_normal.pdf(X, mean=mu_single, cov=sigma_single)
        num = np.dot(pdf,phi_single) 
        
        # Fill the numerator in temp. weight array
        weights[:,i] = num
    
        # Calculate denominator (sum of probabilities * phis)
        denom = denom + num
    
    # Ratio
    denom = np.tile(denom, (4,1))
    w = weights / np.transpose(denom)
    
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    
    return w


def m_step(w, X, mu, sigma, phi, k):
    """
    Computes the M-step of the EM algorithm.
    
    """
    #######################################################################
    # TODO:                                                               #
    # Update all the model parameters as per the M-step of the EM         #
    # algorithm.
    #######################################################################
    
    phi = 1.0/X.shape[0] * np.sum(w, axis=0)
    
    mu = np.dot(np.transpose(w), X) / np.transpose(np.tile(np.sum(w, axis=0), (2,1)))
    
    sigma = np.array(sigma)
    for i in range(k):
        sigma[i] = np.dot(np.multiply(np.transpose(X-np.tile(mu[i,:],(X.shape[0],1))),w[:, i]), (X-np.tile(mu[i,:],(X.shape[0],1))))
  
    sigma = sigma / np.sum(w)
                                          
    # TODO: retransform sigma from array to lists
    sigma = sigma.tolist()  
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    
    return phi, mu, sigma
        
        