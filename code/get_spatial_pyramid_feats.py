import cv2
import numpy as np
from numpy import linalg
import math
from distance import pdist
from feature_extraction import feature_extraction


def get_spatial_pyramid_feats(image_paths, max_level, feature):
    """
    This function assumes that 'vocab_*.npy' exists and
    contains an vocab size x feature vector length matrix 'vocab' where each row
    is a kmeans centroid or visual word. This matrix is saved to disk rather than passed
    in a parameter to avoid recomputing the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path,
    :param max_level: level of pyramid,
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size'), multiplies with
        (1 / 3) * (4 ^ (max_level + 1) - 1).
    """

    vocab = np.load(f'vocab_{feature}.npy')

    vocab_size = vocab.shape[0]
    d = vocab_size * ((4**(max_level+1)-1)//3)
    spatial_pyramid = np.zeros((len(image_paths), d))

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        a, b = img.shape[:2]
        
        for level in range(max_level+1):
            prev_block = (4**level - 1)//3
            two_level = 2**level
            weight = 2**(-max_level) if level == 0 else 2**(-max_level+level-1)
            for x in range(two_level):
                for y in range(two_level):
                    sub_img =  img[x*(a//two_level):(x+1)*(a//two_level),
                                   y*(b//two_level):(y+1)*(b//two_level)]
                    features = feature_extraction(sub_img, feature)
                    dist_matrix = pdist(features, vocab)
                    idxs = np.argmin(dist_matrix, axis=1)
                    hist, _ = np.histogram(idxs, vocab_size, range=(0,vocab_size-1)) 
                    hist = hist*weight
                    num = (x+1)*(y+1)-1 # Number of subimage
                    
                    spatial_pyramid[i,(num+prev_block)*vocab_size:(num+prev_block+1)*vocab_size] = hist
        spatial_pyramid[i,:] = spatial_pyramid[i,:] / np.linalg.norm(spatial_pyramid[i,:])
    print(f"shape is {spatial_pyramid.shape} and value is {spatial_pyramid}")
    return spatial_pyramid
