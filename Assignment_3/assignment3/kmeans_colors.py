from sklearn.cluster import KMeans
import numpy as np


def kmeans_colors(img, k, max_iter=100):
    """
    Performs k-means clustering on the pixel values of an image.
    Used for color-quantization/compression.

    Args:
        img: The input color image of shape [h, w, 3]
        k: The number of color clusters to be computed

    Returns:
        img_cl:  The color quantized image of shape [h, w, 3]

    """

    img_cl = None

    #######################################################################
    # TODO:                                                               #
    # Perfom k-means clustering of on the pixel values of the image img.  #
    #######################################################################
    
    # Convert img to float
    img = np.array(img, dtype=np.float64)/255
    
    width, height, color = img.shape
    
    # Reshape image to a 2D image array
    image_array = np.reshape(img, (width*height, color))
        
    # Clustering
    kmeans = KMeans(n_clusters=k, max_iter=max_iter, random_state=0).fit(image_array)
    
    # Predict color values
    colors = kmeans.predict(image_array)
    
    # Recreate image
    img_cl = np.empty_like(img)
    id = 0;
    for i in range(width):
        for j in range(height):
            img_cl[i][j] = image_array[colors[id]]
            id += 1

    
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return img_cl