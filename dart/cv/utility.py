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
    def get_centroid(cnt):
        M = cv2.moments(cnt)
        div = Utility._div(M['m00'])
        return (int(M['m10']/div), int(M['m01']/div))

    @staticmethod
    def _div(moment):
        if moment == 0:
            return 0.00001
        return moment

    @staticmethod
    def hist_gray(im):
        """
        Return histogram of grayscale image.
        Usage: cv2.imshow("Histogram", hist_gray(img))
        """

        h = np.zeros((300, 256, 3))
        if len(im.shape) != 2:
            print "hist_lines applicable only for grayscale images"
            #print "so converting image to grayscale for representation"
            im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        hist_item = cv2.calcHist([im], [0], None, [256], [0, 256])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        for x, y in enumerate(hist):
            cv2.line(h, (x, 0), (x, y), (255, 255, 255))
        y = np.flipud(h)
        return y