import cv2
import numpy as np


def feature_extraction(img, feature):
    """
    This function computes defined feature (HoG, SIFT) descriptors of the target image.

    :param img: a height x width x channels matrix,
    :param feature: name of image feature representation.

    :return: a number of grid points x feature_size matrix.
    """

    if feature == 'HoG':
        # HoG parameters
        win_size = (32, 32)
        block_size = (32, 32)
        block_stride = (16, 16)
        cell_size = (16, 16)
        nbins = 9
        deriv_aperture = 1
        win_sigma = 4
        histogram_norm_type = 0
        l2_hys_threshold = 2.0000000000000001e-01
        gamma_correction = 0
        nlevels = 64

        args = [win_size, block_size, block_stride, cell_size,
               nbins, deriv_aperture, win_sigma, histogram_norm_type,
               l2_hys_threshold, gamma_correction, nlevels ]
        
        hog = cv2.HOGDescriptor(*args)
        
        des = np.reshape(hog.compute(img), (-1, 36))
        
        return des
        

    elif feature == 'SIFT':

        sift = cv2.SIFT_create()
        
        kp_list = []
        for i in range(0, img.shape[0], 20):
            for j in range(0, img.shape[1], 20):
                kp = cv2.KeyPoint(i, j, 1)
                kp_list.append(kp)

        _, des = sift.compute(img,kp_list)
        
        return des


