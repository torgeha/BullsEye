import math

import cv2
import numpy as np

from utility import Utility
from ocr.learner import DartHelper, DartLearner

from collections import Counter

class Board:

    RED_SCORES = [2, 10,13,18,20,12,14,8,7,3 ,50]
    GREEN_SCORES = [17,15,6,4,1,5,9,11,16,19, 25]
    NR_COLORED_SEGMENTS = 21
    RED_LIMIT = 60
    GREEN_LIMIT = 60

    def __init__(self, red_score=RED_SCORES, green_score=GREEN_SCORES, green_limit=GREEN_LIMIT, red_limit=RED_LIMIT, debug=False):
        if not debug:
            self.learner = DartLearner(samples=DartLearner.SAMPLE_FILENAME, responses=DartLearner.RESPONSE_FILENAME)
        self.green_limit = green_limit
        self.red_limit = red_limit
        self.red_score = red_score
        self.green_score = green_score



    def detect_ellipse(self, image):
        blurred = cv2.GaussianBlur(image,(5,5),0)
        red_mask, green_mask = self._color_difference_segmentation(blurred)
        thresh = self._theshold(blurred)
        red_mask = self._prune_board(thresh, red_mask)
        green_mask = self._prune_board(thresh, green_mask)
        red_scores = self._create_description_areas(red_mask)
        green_scores = self._create_description_areas(green_mask)
        ellipse, approx_hull = self._fit_ellipse(red_scores)
        return ellipse

    def detect(self, image):
        '''
        With the image supplied as argument, a shape description containing the board outline,
        and bulleye coordinates.
        '''

        blurred = cv2.GaussianBlur(image,(5,5),0)
        thresh = self._theshold(blurred)
        #TODO: Make outline and center available as self.
        red_mask, green_mask = self._color_difference_segmentation(blurred)
        cv2.imshow("dfd", thresh)
        cv2.waitKey(1)
        red_mask = self._prune_board(thresh, red_mask)
        green_mask = self._prune_board(thresh, green_mask)
        red_scores = self._create_description_areas(red_mask)
        green_scores = self._create_description_areas(green_mask)

        board_sector = self._identify_board(thresh)
        #for ctn in [board_sector]:
        #    cv2.drawContours(blurred,[cnt],0,255,-1)

        if not self._is_valid(red_scores, green_scores):
            print("Red scores or green scores are not valid")
            return None, None, None

        if  board_sector is None:
            print("Could not find countours, and therefore the board sector")
            return None, None, None

        bounding, approx_hull = self._fit_ellipse([board_sector])
        ellipse, approx_hull = self._fit_ellipse(red_scores, bounding=bounding)
        center = self._identify_bullseye(red_scores, ellipse)
        contours, number_mask, groups = DartHelper.create_number_descriptions(image, ellipse)
        predictions = self.learner.classify_all(number_mask, groups)

        if len(predictions) <20:
            print("To few predictions")
            #TODO: Error correction
            return None, None, None
        predictions = Utility.classification_error_correction(center, predictions)
        red_id = self._id_contours(red_scores,center, ellipse, blurred)
        green_id = self._id_contours(green_scores, center, ellipse, blurred)
        mask = self._create_score_mask(image.shape, ellipse, red_id, green_id, center, predictions)
        return center, ellipse, mask


    def _prune_board(self, thresh, mask):
        return np.array(np.logical_and(thresh, mask)*255, np.uint8)


    def _is_valid(self, red_scores, green_scores):
        return len(red_scores) ==Board.NR_COLORED_SEGMENTS and len(green_scores) >= Board.NR_COLORED_SEGMENTS and len(green_scores) <= Board.NR_COLORED_SEGMENTS+1

    def _create_score_mask(self, size, ellipse, red, green, center, predictions):
        shape = (size[0], size[1])
        mask = np.zeros(shape, np.uint8)
        cv2.ellipse(mask, ellipse, 100 ,thickness=-1)
        g = self._get_scores_for_contours(green[0], predictions, center)
        g2 = self._get_scores_for_contours(green[1], predictions, center)
        r = self._get_scores_for_contours(red[0], predictions, center)
        r2 = self._get_scores_for_contours(red[1], predictions, center)
        mask = self._draw_sectors(mask, green[0], center, g)

        mask = self._draw_sectors(mask, red[0], center, r)
        self._draw_special(mask, green, (g,g2, 25))
        self._draw_special(mask,red, (r,r2, 50))
        for i,cnt in enumerate(green[0]):
            c = cnt[2]
            cv2.putText(mask,str(g[i]),(Utility.get_centroid(c)),0,1,255, thickness=1)
        for i,cnt in enumerate(red[0]):
            c = cnt[2]
            cv2.putText(mask,str(r[i]),(Utility.get_centroid(c)),0,1,255, thickness=1)
        return mask

    def _get_scores_for_contours(self, contours, predictions, center):
        v = []
        scores = []
        for p in predictions:
            x,y,w,h = p[0]
            v.append(np.array([center[0]-x, center[1]-y]))

        for sector in contours:
            c = sector[2]

            x,y = Utility.get_centroid(c)
            u = np.array([center[0]-x, center[1]-y])
            n = Utility.find_closest(v, u)
            scores.append(predictions[n][1])
        return scores

    def _draw_sectors(self, mask, sectors, center, score):
        '''
        Use the outer ring to calculate extreme points. These points are averaged to e1 and e2, which
        combined with the center points becomes a sector.
        '''
        for i in range(len(sectors)):
            t = sectors[i][2]
            c =  sectors[i][0]
            #TODO: Avoid convexHUll?
            #Use orientation somehow?
            rect = cv2.minAreaRect(t)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box = np.vstack([box, center])
            box = cv2.convexHull(box)
            cv2.fillConvexPoly(mask, box, score[i])

        return mask

    def _draw_special(self, mask, score_areas, scores):
        outer = score_areas[0]
        inner = score_areas[1]
        center = score_areas[2]

        for i in range(len(outer)):
            cv2.drawContours(mask, [outer[i][2]], -1, 2*scores[0][i], thickness=-1)

        for j in range(len(inner)):
            cv2.drawContours(mask, [inner[j][2]], -1, 3*scores[1][j], thickness=-1)

        for k in range(len(center)):
            cv2.drawContours(mask, [center[k][2]], -1, scores[2], thickness=-1)

    def _extract_edges(self, image):
        return cv2.Canny(image,100,200)

    def _create_description_areas(self, mask):
        #TODO: better way of combining
        mask = Utility.expand(mask)

        #TODO: Remove countours that should not be there. 21, and 22 countours not more or less.
        img,contours,hierarchy = cv2.findContours(mask,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        #TODO:Combine close ones, and prune noise outside.
        #print(len(contours))
        return contours

    def _id_contours(self, contours, center, ellipse, blurred):
        if(len(contours)>22):
            #TODO: Make more robust than this
            raise Exception("TOO many score areas identified!!!")
        outer_area = []
        inner_area = []
        short_ellipse = Utility.scale_ellipse(ellipse, 0.75)
        center_ellipse = Utility.scale_ellipse(ellipse, 0.35)

        center_area = []
        for cnt in contours:
            x,y = Utility.get_centroid(cnt)
            x_v = center[0] -x
            y_v = center[1] - y
            #TODO: Angle neccesary???

            dist = math.sqrt(x_v**2+ y_v**2)
            inside = Utility.inside_ellipse((x,y), short_ellipse)
            is_center = Utility.inside_ellipse((x,y), center_ellipse)
            a = 0 #Removed angle, but not from tuple
            description = (a, dist, cnt)
            if not inside:
                outer_area.append(description)
            elif inside and not is_center:
                inner_area.append(description)
            else:
                center_area.append(description)

        return (outer_area, inner_area, center_area)


    def _identify_bullseye(self, descriptions, ellipse):
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

    def _fit_ellipse(self, contours, bounding=None):
        if bounding:
            contours = self._prune(contours, bounding)
        cont = np.vstack(ctn for ctn in contours)
        hull =  cv2.convexHull(cont)
        ellipse = cv2.fitEllipse(hull)
        return ellipse, hull

    def _prune(self, contours, bounding):
        prune_indices = []
        for i,cnt in enumerate(contours):

            if not Utility.inside_ellipse(Utility.get_centroid(cnt), bounding):
                prune_indices.append(i)
        return np.delete(contours, prune_indices)

    def _identify_board(self, thresh):
        #TODO: move to identify, use threshold elsewhere
        img,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) <= 0:
            return None
        max_contour = max(contours, key=lambda c: cv2.arcLength(c, True))


        occurances = {}
        #for h in hierarchy[0][0]:
        #    if h[3] not in occurances:
        #        occurances[h[3]] = 1
        #    occurances[h[3]] += 1
        #cv2.drawContours(image, max_contour, -1, (255,0,0), 2)
        #cv2.imshow("ttt", image)
        #cv2.waitKey(-1)
        return max_contour

    def _theshold(self, image):
        blurred = cv2.GaussianBlur(image,(25,25),0)
        grey = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        ret,thresh = cv2.threshold(grey,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #thresh = Utility.expand(thresh, kernel=np.ones((7,7), np.uint8))
        des = cv2.bitwise_not(thresh)
        #Hole filling
        #TODO: Generalize , so findcountours does not get called like 10 times.
        img,contours,hierarchy = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(des,[cnt],0,255,-1)
        kernel = Utility.circular_kernel(40)
        des  = Utility.remove_bw_noise(des, kernel=kernel)
        return des

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

        kernel = np.ones((3,3), np.uint8)
        red = Utility.remove_bw_noise(red, kernel=kernel)
        red = Utility.expand(red)
        green = Utility.remove_bw_noise(green, kernel=kernel)
        green = Utility.expand(green)
        return red, green
