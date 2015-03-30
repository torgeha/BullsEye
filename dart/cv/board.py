import cv2
import numpy as np
from utility import Utility
import math

class Board:
    DEFAULT_RED_SCORES = [10,2,3,7,8,14,12,20, 18, 13, 50]
    DEFAULT_GREEN_SCORES = [6,15,17,19,16,11,9,5,1,4, 25]
    NR_COLORED_SEGMENTS = 21

    def __init__(self, red_score=DEFAULT_RED_SCORES, green_score=DEFAULT_GREEN_SCORES):
        self.green_limit = 60
        self.red_limit = 60
        self.red_score = red_score
        self.green_score = green_score

    def detect(self, image):
        '''
        With the image supplied as argument, a shape description containing the board outline,
        and bulleye coordinates.
        '''

        blurred = cv2.GaussianBlur(image,(5,5),0)
        #TODO: Make outline and center available as self.
        red_mask, green_mask = self._color_difference_segmentation(blurred)
        red_scores = self._create_description_areas(red_mask)
        green_scores = self._create_description_areas(green_mask)
        if not self._is_valid(red_scores, green_scores):
            #TODO: error handling. Remove or fix stuff
            return None, None, None
        ellipse, approx_hull = self._fit_ellipse(red_scores)
        center = self._identify_bullseye(red_scores, ellipse)
        orientation = self._orientation(blurred, center, ellipse)
        print(orientation)
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        cv2.circle(blurred, center, 5, (255, 0,0), 1)
        cv2.drawContours(blurred, green_scores, -1, (0, 255,0), 1)
        cv2.imshow("nu",blurred)
        cv2.waitKey(-1)

        red_id = self._id_contours(red_scores,center, ellipse)
        green_id = self._id_contours(green_scores, center, ellipse)
        mask = self._create_score_mask(image.shape, ellipse, red_id, green_id, center, orientation)
        return center, ellipse, mask

    def _is_valid(self, red_scores, green_scores):
        return len(red_scores) ==Board.NR_COLORED_SEGMENTS and len(green_scores) == Board.NR_COLORED_SEGMENTS+1


    def _orientation(self, blurred, center, ellipse):
        #TODO: Optional orientation measure. Should work without this working. As long as
        #Camera is the right way
        number_mask = self._outline_segmentation(blurred)
        number_mask = cv2.ellipse(number_mask, ellipse, (0,0,0), thickness=-1)
        wide_ellipse = (ellipse[0], (ellipse[1][0]*1.25,ellipse[1][1]*1.25) , ellipse[2])
        mask = np.zeros(number_mask.shape, np.uint8)

        cv2.ellipse(mask, wide_ellipse, 1, thickness=-1)
        number_mask = cv2.multiply(number_mask, mask)
        number_mask = cv2.erode(number_mask,  np.ones((2,2),np.uint8))

        img,contours,hierarchy = cv2.findContours(number_mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        children = []
        w = wide_ellipse[1][0]-ellipse[1][0]
        lowest = float('inf')
        eleven = None
        for c in range(len(contours)-1):
            #TODO: Reduce double calculations
            c1 = contours[c]
            c2 = contours[c+1]
            cc1 = Utility.get_centroid(c1)
            cc2 = Utility.get_centroid(c2)
            d = abs(cc1[0] - cc2[0]) + abs(cc1[1] - cc2[1])
            m1 = (cv2.contourArea(c1)/cv2.arcLength(c1, True))
            m2 = (cv2.contourArea(c2)/cv2.arcLength(c2, True))

            if d < w and m1 + m2< lowest:
                #TODO: less than 30 not robust!!
                lowest = m1 + m2
                eleven = c1, c2
        x,y = Utility.get_centroid(eleven[0])
        return Utility.angle(center, x, y)

    def _create_score_mask(self, size, ellipse, red, green, center, orientation):
        shape = (size[0], size[1])
        mask = np.zeros(shape, np.uint8)
        cv2.ellipse(mask, ellipse, 100 ,thickness=-1)
        mask = self._draw_sectors(mask, green[0], center, self.green_score)
        mask = self._draw_sectors(mask, red[0], center, self.red_score)
        self._draw_special(mask, green, self.green_score, orient=orientation)
        self._draw_special(mask,red, self.red_score, orient=orientation)
        return mask

    def _draw_sectors(self, mask, sectors, center, score):
        '''
        Use the outer ring to calculate extreme points. These points are averaged to e1 and e2, which
        combined with the center points becomes a sector.
        '''
        for i in range(len(sectors)):
            t = sectors[i][2]
            v =  sectors[i][0]
            #TODO: Avoid convexHUll?
            #Use orientation somehow?
            rect = cv2.minAreaRect(t)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box = np.vstack([box, center])
            box = cv2.convexHull(box)
            cv2.fillConvexPoly(mask, box, score[int(math.floor(i/2))])
        return mask

    def _draw_special(self, mask, score_areas, scores, orient=0):
        outer = score_areas[0]
        inner = score_areas[1]
        center = score_areas[2]
        print(len(outer))
        print(len(inner))
        for i in range(len(outer)):
            cv2.drawContours(mask, [outer[i][2]], -1, 2*scores[i], thickness=-1)
        for i in range(len(inner)):

            cv2.drawContours(mask, [inner[i][2]], -1, 3*scores[i], thickness=-1)
        cv2.drawContours(mask, [center[2]], -1, scores[-1], thickness=-1)

    def _extract_edges(self, image):
        return cv2.Canny(image,100,200)

    def _create_description_areas(self, mask):
        #TODO: better way of combining
        mask = Utility.expand(mask)
        #TODO: Remove countours that should not be there. 21, and 22 countours not more or less.
        img,contours,hierarchy = cv2.findContours(mask,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        #TODO:Combine close ones
        print(len(contours))
        return contours

    def _id_contours(self, contours, center, ellipse):
        if(len(contours)>22):
            #TODO: Make more robust than this
            raise Exception("TOO many score areas identified!!!")
        outer_area = []
        inner_area = []

        c = None
        for cnt in contours:
            x,y = Utility.get_centroid(cnt)
            x_v = center[0] -x
            y_v = center[1] - y

            dist = math.sqrt(x_v**2+ y_v**2)
            a = Utility.angle(center, x, y)
            if dist > 130:
                #TODO: ellipse radius

                outer_area.append((a,dist , cnt ))
            elif dist > 10:
                inner_area.append((a,dist,cnt))
            else:
                c = (a,dist , cnt)
        outer_area.sort(key=lambda k: k[0])
        inner_area.sort(key=lambda k: k[0])

        return (outer_area, inner_area, c)


    def _identify_bullseye(self, descriptions, ellipse):

        #avg_x, avg_y = Utility.get_avg_pos(descriptions)
        #Center of ellipse might be some off the true center.
        #Find contour that match center best and return it
        avg_x, avg_y = ellipse[0]
        min_dist = float("inf")
        center = None

        for ctn in descriptions:
            x, y = Utility.get_centroid(ctn)
            dist = (avg_x - x)**2 + (avg_y-y)**2
            if dist < min_dist:
                center = (x,y)
                min_dist = dist
        return center

    def _fit_ellipse(self, contours):
        if(len(contours)>4):
            cont = np.vstack(ctn for ctn in contours)
            hull =  cv2.convexHull(cont)
            ellipse = cv2.fitEllipse(hull)
            return ellipse, hull
        return None

    def _color_difference_segmentation(self, image):
        b,g,r = cv2.split(image)

        #Filter out all r/g pairs with a lower difference than red and green limit. Black and white gone
        grey_diff = cv2.subtract(r, g, dtype=cv2.CV_16S)
        red = np.greater(grey_diff, self.red_limit)
        green = np.less(grey_diff, -self.green_limit)

        #Change datatype and values from 0-1 range to 0-255 range
        red = Utility.convert_to_cv(red)
        green = Utility.convert_to_cv(green)

        #Remove noise
        red = Utility.remove_bw_noise(red)
        red = Utility.expand(red)
        green = Utility.remove_bw_noise(green)
        green = Utility.expand(green)
        return red, green

    def _outline_segmentation(self, image):
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret,thresh = cv2.threshold(grey,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh
