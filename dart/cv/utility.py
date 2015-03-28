import numpy as np
import cv2

class Utility:

    @staticmethod
    def convert_to_cv(normalized_matrix):
        image = np.array(normalized_matrix, dtype='uint8')
        image = image*255
        return image

    @staticmethod
    def remove_bw_noise(noisy_image, kernel=None):
        if not kernel:
            kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(noisy_image,kernel,iterations = 1)
        dilation = cv2.dilate(erosion,kernel,iterations = 1)
        return dilation