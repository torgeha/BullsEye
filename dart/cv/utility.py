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

    @staticmethod
    def get_avg_pos(contours):
        avg_pos_x = 0
        avg_pos_y = 0
        for cnt in contours:
            M = cv2.moments(cnt)
            avg_pos_x += int(M['m10']/M['m00'])
            avg_pos_y += int(M['m01']/M['m00'])
        avg_pos_x = avg_pos_x/len(contours)
        avg_pos_y = avg_pos_y/len(contours)
        return (avg_pos_x, avg_pos_y)

    @staticmethod
    def get_centroid(cnt):
        M = cv2.moments(cnt)
        return (int(M['m10']/M['m00']), int(M['m01']/M['m00']))