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
        outline = self._outline_segmentation(blurred)
        red_mask, green_mask = self._color_difference_segmentation(blurred)
        cv2.imshow("red",red_mask)
        cv2.imshow("green",green_mask)
        cv2.imshow("b", blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def _extract_edges(self, image):
        return cv2.Canny(image,100,200)

    def _create_score_areas(self, mask):
        #TODO: describe all areas, assign id for each description
        #TODO: description should have x,y and size and id.
        pass

    def _identify_bullseye(self, descriptions):
        #TODO: Find center of mass for red descriptions and return it as bullseye
        pass

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