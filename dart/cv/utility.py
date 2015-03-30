import numpy as np
import cv2
import math

class Utility:

    @staticmethod
    def convert_to_cv(normalized_matrix):
        image = np.array(normalized_matrix, dtype='uint8')
        image = image*255
        return image

    @staticmethod
    def remove_bw_noise(noisy_image, kernel=None):
        if kernel == None:
            kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(noisy_image,kernel,iterations = 1)
        dilation = cv2.dilate(erosion,kernel,iterations = 1)
        return dilation

    @staticmethod
    def get_avg_pos(contours):
        if(len(contours)>0):
            avg_pos_x = 0
            avg_pos_y = 0
            for cnt in contours:
                M = cv2.moments(cnt)
                div = Utility._div(M['m00'])
                avg_pos_x += int(M['m10']/div)
                avg_pos_y += int(M['m01']/div)

            avg_pos_x = avg_pos_x/len(contours)
            avg_pos_y = avg_pos_y/len(contours)
            return (avg_pos_x, avg_pos_y)
        else:
            return (-1, -1)

    @staticmethod
    def to_degree(angle):
        angle = angle * 360 / (2*math.pi);
        if (angle < 0):
            angle = angle + 360;


    @staticmethod
    def get_centroid(cnt):
        M = cv2.moments(cnt)
        div = Utility._div(M['m00'])
        return (int(M['m10']/div), int(M['m01']/div))

    @staticmethod
    def expand(mask ,kernel=None):
        if not kernel:
            kernel = np.ones((3,3),np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    @staticmethod
    def angle( center, x,y):
        return math.atan2(center[0]-x, center[1]-y)

    @staticmethod
    def _div(moment):
        if moment == 0:
            return 0.00001
        return moment