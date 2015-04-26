import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from collections import deque

class Utility:
    SCORES = deque([17,2,15,10,6,13,4,18,1,20,5,12,9,14,11,8,16,7,19,3])
    DEBUG = False
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
    def circular_kernel(radius):
        circle = np.zeros((radius*2,radius*2), np.uint8)
        return cv2.circle(circle,(radius,radius),radius,1,thickness=-1)

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
    def find_closest(v_list, u):
        #TODO: List comprehension
        p = float('inf')
        closest = 0
        for i, v in enumerate(v_list):

            temp = np.dot(v, u) / (np.linalg.norm(v)*np.linalg.norm(u))
            if np.allclose(temp, -1):
                angle = math.pi
            elif np.allclose(temp, -1):
                angle = 0
            else:
                angle = math.acos(temp)

            if angle < p:
                closest = i
                p = angle
        return closest

    @staticmethod
    def classification_error_correction(center, classifications):
        #Will sort based on proximity
        #Ensure that there are only 20 objects
        #Error correct based on order
        if len(classifications) != 20:
            print(len(classifications))
            if (len(classifications))<20:
                print("Less than 20 classifications")
                Utility.DEBUG = True
            if len(classifications) > 20:
                print("More than 20 classifications")
            while(len(classifications)) > 20:
                classifications = sorted(classifications, key=lambda i: i[1], reverse=True)
                popped = classifications.pop()
                print(popped)
                print(len(classifications))

        s = sorted(classifications, key=lambda i: Utility.angle(center, i[0][0], i[0][1]))
        #TODO: If less than 20 found. Use the few left and rebuild using angles and vectors
        score = Utility.SCORES
        max_correct = 0
        best = 0
        for i in range(len(score)):
            correct = sum([p == c[1] for p,c in zip(score, s)])
            if correct > max_correct:
                max_correct = correct
                best = i
            score.rotate()
        score = Utility.SCORES
        for i in range(len(score)):
            v = score[(i-best)%len(score)]
            s[i] = (s[i][0], v)
        return s


    @staticmethod
    def get_centroid(cnt):
        M = cv2.moments(cnt)
        div = Utility._div(M['m00'])
        return (int(M['m10']/div), int(M['m01']/div))

    @staticmethod
    def expand(mask ,kernel=None):
        if kernel == None:
            kernel = np.ones((3,3),np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    @staticmethod
    def inside_ellipse(point, ellipse):
        px, py = point
        h, k = ellipse[0]
        rx, ry = ellipse[1]
        angle = ellipse[2]
        x, y =np.dot(np.array([px-h, py-k]), Utility.rot_matrix(-angle))
        a = rx * 0.5
        b = ry * 0.5
        return (x**2/a**2) + (y**2/b**2) <= 1

    @staticmethod
    def rot_matrix(theta):
        return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]);

    @staticmethod
    def ellipse_area(ellipse):
        w, h = ellipse[1][0],ellipse[1][1]
        return math.pi * w/2 * h/2

    @staticmethod
    def scale_ellipse(ellipse, factor):
        #ellipse = ((center),(width,height of bounding rect), angle)
        return (ellipse[0], (ellipse[1][0]*factor,ellipse[1][1]*factor) , ellipse[2])

    @staticmethod
    def angle( center, x,y):
        return math.atan2(center[0]-x, center[1]-y)

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

    def show_value_plot(image):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image, cmap=cm.jet, interpolation='nearest')
        numrows, numcols = image.shape
        def format_coord(x, y):
            col = int(x+0.5)
            row = int(y+0.5)
            if col>=0 and col<numcols and row>=0 and row<numrows:
                z = image[row,col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f'%(x, y)

        ax.format_coord = format_coord
        plt.show()
