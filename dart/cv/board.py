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
        grid = self._extract_edges(blurred)
        red_mask, green_mask = self._color_difference_segmentation(blurred)
        red_scores = self._create_score_areas(red_mask)
        green_scores = self._create_score_areas(green_mask)
        if not self._is_valid(red_scores, green_scores):
            #TODO: error handling. Remove or fix stuff
            return None, None, None
        ellipse, approx_hull = self._fit_ellipse(red_scores)
        cv2.ellipse(grid, ellipse, (0,0,255))
        center = self._identify_bullseye(red_scores)
        red_id = self._id_contours(red_scores,center)
        green_id = self._id_contours(green_scores, center)
        mask = self._create_score_mask(image.shape, ellipse, red_id, green_id, center)
        return center, ellipse, mask

    def _is_valid(self, red_scores, green_scores):
        return len(red_scores) ==Board.NR_COLORED_SEGMENTS and len(green_scores) == Board.NR_COLORED_SEGMENTS+1

    def _create_score_mask(self, size, ellipse, red, green, center):
        shape = (size[0], size[1])
        mask = np.zeros(shape, np.uint8)
        cv2.ellipse(mask, ellipse, 100 ,thickness=-1)
        mask = self._draw_sectors(mask, green, red, center, self.green_score, red=True)
        mask = self._draw_sectors(mask, red,green, center, self.red_score, red=False)
        self._draw_special(mask, green, self.green_score)
        self._draw_special(mask,red, self.red_score)
        return mask

    def _draw_sectors(self, mask, sectors, adjusters, center, score, red=True):
        '''
        Use the outer ring to calculate extreme points. These points are averaged to e1 and e2, which
        combined with the center points becomes a sector.
        '''
        previous = 0
        next = 2
        if red:
            previous = -2
            next = 0
        for i in range(1,len(sectors)-1, 2):
            t = sectors[i][2]
            g1 = adjusters[(i+previous)%(len(sectors)-1)][2]
            g2 = adjusters[(i+next)%(len(sectors)-1)][2]
            v =  sectors[i][0]

            if  v> 2.5 or v<-2.0 or (v>-0.5 and v<1.2):
                #If angle is v, means top and bottom extreme points has to be used as e1 and e2
                topmost = np.array(t[t[:,:,1].argmin()][0])
                g_bottom = np.array(g1[g1[:,:,1].argmax()][0])
                g_top = np.array(g2[g2[:,:,1].argmin()][0])
                bottommost = np.array(t[t[:,:,1].argmax()][0])

                e1 = (topmost + g_bottom) / 2
                e2 = (bottommost + g_top) / 2
            else:
                #If not angle fit the if statement, left and right extreme points has to be used as e1, and e2
                g_right = np.array(g2[g2[:,:,0].argmax()][0])
                g_left = np.array(g1[g1[:,:,0].argmin()][0])
                leftmost = np.array(t[t[:,:,0].argmin()][0])
                rightmost = np.array(t[t[:,:,0].argmax()][0])
                e1 = (leftmost + g_right) / 2
                e2 = (rightmost + g_left) / 2
            cv2.fillConvexPoly(mask, np.array([e1, e2, center]), score[int(math.floor(i/2))])
        return mask

    def _draw_special(self, mask, score_areas, scores):
        #TODO: Morph erea a bit out.
        for i in range(len(score_areas)-1):
            r = score_areas[i]
            score = (3- i%2) * self.red_score[int(math.floor(i/2))]
            cv2.drawContours(mask, [r[2]], -1, score, thickness=-1)
        cv2.drawContours(mask, [score_areas[-1][2]], -1, scores[-1], thickness=-1)

    def _extract_edges(self, image):
        return cv2.Canny(image,100,200)

    def _create_score_areas(self, mask):
        #TODO: Remove countours that should not be there. 21, and 22 countours not more or less.
        img,contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        print(len(contours))
        return contours

    def _id_contours(self, contours, center):
        if(len(contours)>22):
            #TODO: Make more robust than this
            raise Exception("TOO many score areas identified!!!")
        sorted_contours = []
        c = None
        for cnt in contours:
            x,y = Utility.get_centroid(cnt)
            x_v = center[0] -x
            y_v = center[1] - y
            dist = x_v**2+ y_v**2
            if math.sqrt(dist) > 10:
                sorted_contours.append((math.atan2(y_v,x_v),dist , cnt))
            else:
                c = (math.atan2(y_v,x_v),dist , cnt)
        sorted_contours.sort(key=lambda k: (round(k[0], 1), k[1]))
        sorted_contours.append(c)
        return sorted_contours

    def _angle(self, a1, a2):
        return np.arccos((a1 * a2) / (np.abs(a1) * np.abs(a2)))

    def _identify_bullseye(self, descriptions):
        #TODO: MAKE MORE ROBUST. AVG_pos not that great!
        avg_x, avg_y = Utility.get_avg_pos(descriptions)
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
        #TODO: Use more robust method than thresholding
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret,thresh = cv2.threshold(grey,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh
