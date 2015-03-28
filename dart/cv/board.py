import cv2
import numpy as np
from utility import Utility

class Board:

    def __init__(self):
        self.green_limit = 60
        self.red_limit = 60


    def detect(self, image):
        '''
        With the image supplied as argument, a shape description containing the board outline,
        and bulleye coordinates.
        '''
        #TODO: Find outline
        #TODO: Find bullseye center
        #TODO: Create a shape dscription and return it
        blurred = cv2.GaussianBlur(image,(5,5),0)
        #outline = self._outline_segmentation(blurred)
        red_mask, green_mask = self._color_difference_segmentation(blurred)
        red_scores = self._create_score_areas(red_mask)
        green_scores = self._create_score_areas(green_mask)
        center = self._identify_bullseye(red_scores)
        outline = self._fit_ellipse(red_scores)
        cv2.circle(blurred, center, 5, (0, 255,100), thickness=2)
        #cv2.imshow("red",red_mask)
        #cv2.imshow("green",green_mask)
        cv2.imshow("abc", blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def _extract_edges(self, image):
        return cv2.Canny(image,100,200)

    def _create_score_areas(self, mask):
        #TODO: describe all areas, assign id for each description
        #TODO: description should have x,y and size and id.
        #TODO: Remove countours that should not be there. 21, and 22 countours not more or less.
        img,contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        print(len(contours))

        cv2.imshow("C", self._visualize(contours))
        return contours

    def _visualize(self, contours):
        #unified = []
        #cont = np.vstack(ctn for ctn in contours)
        #hull =  cv2.convexHull(cont)
        #unified.append(hull)
        blank_image = np.zeros((480,640,3), np.uint8)
        l = [Utility.get_centroid(ctn) for ctn in contours]
        for c in l:
            cv2.circle(blank_image, c, 5, (200, 200 ,0))
        cv2.circle(blank_image, Utility.get_avg_pos(contours), 5, (0, 200, 200))
        cv2.imshow("dkjfjd", blank_image)
        #cv2.drawContours(blank_image,unified,-1,(0,255,0),2)
        #cv2.drawContours(blank_image, contours, -1, (0,0,255), 1)
        return blank_image

    def _identify_bullseye(self, descriptions):
        avg_x, avg_y = Utility.get_avg_pos(descriptions)
        min_dist = float("inf")
        center = None
        self._visualize(descriptions)
        for ctn in descriptions:
            x, y = Utility.get_centroid(ctn)
            dist = (avg_x - x)**2 + (avg_y-y)**2
            if dist < min_dist:
                center = (x,y)
                min_dist = dist
        return center

    def _fit_ellipse(self, contours):
        cont = np.vstack(ctn for ctn in contours)
        hull =  cv2.convexHull(cont)
        return hull

    def _color_difference_segmentation(self, image):
        b,g,r = cv2.split(image)

        #Opencv use usigned i bit integers 0-255. But need negative values. Use int 16 instead
        g = np.array(g, dtype="int16")
        r = np.array(r, dtype="int16")

        #Filter out all r/g pairs with a lower difference than red and green limit. Black and white gone
        grey_diff = r-g
        red = np.greater(grey_diff, self.red_limit)
        green = np.less(grey_diff, -self.green_limit)

        #Change datatype and values from 0-1 range to 0-255 range
        red = Utility.convert_to_cv(red)
        green = Utility.convert_to_cv(green)

        #Remove noise
        red = Utility.remove_bw_noise(red)
        green = Utility.remove_bw_noise(green)
        return red, green

    def _outline_segmentation(self, image):
        #TODO: Use more robust method than thresholding
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret,thresh = cv2.threshold(grey,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh
