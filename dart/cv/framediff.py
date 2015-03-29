
import cv2
import numpy as np

from utility import Utility

class Detector:
    """
    Subframe, used to determine if change in frame is due to arrow, camera change or other.
    """

    def __init__(self, xmin, ymin, width, height):
        self.xmin = xmin
        self.ymin = ymin
        self.widt = width
        self.height = height

    def includes(self, pix_x, pix_y):
        """
        Return true if given pixel is inside this detector.
        """
        # TODO: expand to take range of pixels?
        pass


def classify_change(base_frame, new_frame, percent_threshold, max_change):
    """
    Return 0: Nothing
    Return 1: New arrow
    Return 2: Camera change

    Returns tuple of (percent, value)
    """

    # TODO: expand to use detectors based on dart board location

    thresh = _process_change(base_frame, new_frame, 50)

    # Sum of pixels
    pixel_sum = cv2.sumElems(thresh)[0] # TODO: more efficient way to do this?

    # Percent of frame that differs
    change_percent = (pixel_sum / max_change) * 100

    # TODO: tune these parameters to dart board
    if change_percent < percent_threshold: # Change is under percent_threshold, nothing changed
        return (change_percent, 0)
    elif change_percent < 1.5: # Change is more than thresh, less than 1.5 --> arrow
        return (change_percent, 1)
    else: # More has changed --> camera
        return (change_percent, 2)

def _process_change(base_img, new_img, thresh):
    """
    Takes two grayvalue images. Computes diff, blurs and global threshold. Returns bw image.
    This is used to find how much changed.
    """

    # Compute diff
    diff_img = _compute_diff(base_img, new_img)

    # cv2.imshow("diff1", diff_img)

    # Gaussian blur
    blur = cv2.GaussianBlur(diff_img, (5, 5), 0)
    # cv2.imshow("blur", blur)

    # Global threshold
    ret, thresh_img = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY) # TODO: thresh value as parameter?

    cv2.imshow("processed", thresh_img)

    return thresh_img

def find_arrow(base_img, arrow_img):
    """
    Both params are gray  valua images.
    """

    # Compute diff
    diff_img = _compute_diff(base_img, arrow_img)

    # Some morphology and thresholding to isolate arrow further
    isolated_arrow_img = _isolate_arrows(diff_img)
    cv2.imshow("isolatedArrow", isolated_arrow_img)

    # Find coordinates within picture based on isolated arrows
    x, y = _locate_arrow(isolated_arrow_img)


def _isolate_arrows(diff_img):
    """
    Takes a grayscale img, computes diff and some morphology.
    Returns bw-img with arrow contours.
    """

    strel = np.ones((3, 3), np.uint8)
    open = cv2.morphologyEx(diff_img, cv2.MORPH_OPEN, strel)
    # cv2.imshow("open2", open)

    blur = cv2.GaussianBlur(open, (5, 5), 0)
    # cv2.imshow("blur3", blur)

    ret, thresh_img = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # cv2.imshow("thresh4", thresh_img)

    strel2 = np.ones((5, 5), np.uint8)
    open_thresh = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, strel2)
    # cv2.imshow("open_thresh5", open_thresh)

    return open_thresh

def _locate_arrow(bw_img):
    """
    Takes a bw-img with arrow contours.
    Return coordinates of arrows.
    """
    return 0, 0

def _compute_diff(base_img, new_img):
    # Compute diff
    diff_img = cv2.subtract(new_img, base_img, dtype=cv2.CV_16S)
    diff_img = cv2.convertScaleAbs(diff_img)
    return diff_img

# TODO: What change happened? # Arrow? --> isolate and locate. # Camchange --> return that.

# Testing

# base = cv2.imread("C:\Users\Torgeir\Desktop\\base.jpg")
# d1 = cv2.imread("C:\Users\Torgeir\Desktop\d1.jpg")
# d2 = cv2.imread("C:\Users\Torgeir\Desktop\d2.jpg")
#
# width = len(base[0])
# height = len(base)
# nof_pixels = width * height
# max_change = nof_pixels * 255
#
# grayb = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
# d1g = cv2.cvtColor(d1, cv2.COLOR_BGR2GRAY)
# d2g = cv2.cvtColor(d2, cv2.COLOR_BGR2GRAY)
#
# # cv2.imshow("base", grayb)
# # cv2.imshow("d1", d1g)
# # cv2.imshow("d2", d2g)
# # diff1 = process_change(grayb, d1g, 50)
# # diff2 = process_change(grayb, d2g, 50)
# #
# # cv2.imshow("diff1", diff1)
#
# print classify_change(grayb, d1g, 1.0, max_change)
# print classify_change(grayb, d2g, 1.0, max_change)
#
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
