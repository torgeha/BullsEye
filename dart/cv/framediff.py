
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

    # Dilating to merge blobs belonging to the same arrow together
    strel3 = np.ones((5, 5), np.uint8)
    open_thresh = cv2.morphologyEx(open_thresh, cv2.MORPH_DILATE, strel3)

    return open_thresh

def _locate_arrow(bw_img):
    """
    Takes a bw-img with arrow contours.
    Return coordinates of arrows.
    """

    # First merge arrows that are separated
    cv2.imshow("bw", bw_img)


    image, contours, hierarchy = cv2.findContours(bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bw_img = cv2.drawContours(bw_img, contours, -1, 255)

    cv2.imshow("cnt", bw_img)

    # (x,y),(MA,ma),angle = cv2.fitEllipse(contours[0])
    #
    # print x, y
    # print MA, ma
    # print angle

    # Find centroid of every cnt and point firthest away from it
    centers = []
    for cnt in contours:

        M = cv2.moments(cnt)
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])
        centers.append((centroid_x, centroid_y))
        # cv2.circle(bw_img, (centroid_x, centroid_y), 2, 255)

    # TODO: Find the point furthest from centroid!!

    # find extreme points
    index = 0
    for cnt in contours:
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

        # cv2.circle(bw_img, leftmost, 2, 255)
        # cv2.circle(bw_img, rightmost, 2, 255)
        # cv2.circle(bw_img, topmost, 2, 255)
        # cv2.circle(bw_img, bottommost, 2, 255)

        print leftmost, rightmost, topmost, bottommost
        print centers[index]

        dist = np.linalg.norm(leftmost - centers[index])

        index += 1



    cv2.imshow("mass", bw_img)

    rows, cols = bw_img.shape[:2]
    # print rows, cols

    # loop through contours
    # for i in range(len(contours)):
    #     [vx, vy, x, y] = cv2.fitLine(contours[i], cv2.DIST_L2, 0, 0.01, 0.01)
    #     # print vx, vy, x, y
    #     lefty = int((-x * vy / vx) + y)
    #     righty = int(((cols - x) * vy / vx) + y)
    #     bw_img = cv2.line(bw_img, (cols - 1, righty), (0, lefty), 255, 2)
    #
    # cv2.imshow("line", bw_img)

    return 0, 0

def _compute_diff(base_img, new_img):
    # Compute diff
    diff_img = cv2.subtract(new_img, base_img, dtype=cv2.CV_16S)
    diff_img = cv2.convertScaleAbs(diff_img)
    return diff_img

# TODO: What change happened? # Arrow? --> isolate and locate. # Camchange --> return that.

# Testing

base = cv2.imread("C:\Users\Torgeir\Desktop\\base.jpg")
d1 = cv2.imread("C:\Users\Torgeir\Desktop\d1.jpg")
d2 = cv2.imread("C:\Users\Torgeir\Desktop\d2.jpg")
#
# width = len(base[0])
# height = len(base)
# nof_pixels = width * height
# max_change = nof_pixels * 255
#
grayb = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
d1g = cv2.cvtColor(d1, cv2.COLOR_BGR2GRAY)
d2g = cv2.cvtColor(d2, cv2.COLOR_BGR2GRAY)

find_arrow(grayb, d2g)

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
cv2.waitKey(0)
cv2.destroyAllWindows()
