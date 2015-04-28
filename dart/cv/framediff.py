
import cv2
import numpy as np

from utility import Utility

def classify_change(base_frame, new_frame, percent_threshold, max_change):
    """
    Return 0: Nothing
    Return 1: New arrow
    Return 2: Camera change

    Returns tuple of (percent, value)
    """

    # New shit, based on bounding box
    # cv2.imshow("base", base_frame)
    # cv2.imshow("new_frame", new_frame)

    thresh = _process_change(base_frame, new_frame, 70)

    # Sum of pixels
    pixel_sum = cv2.sumElems(thresh)[0] # TODO: more efficient way to do this?

    # Percent of frame that differs
    change_percent = (pixel_sum / max_change) * 100

    # TODO: tune these parameters to dart board
    if change_percent < percent_threshold: # Change is under percent_threshold, nothing changed
        return (change_percent, 0)
    elif change_percent < 7: # Change is more than thresh, less than 1.5 --> arrow
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
    # cv2.imshow("blur2", blur)

    # Global threshold
    ret, thresh_img = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY) # TODO: thresh value as parameter?

    # cv2.imshow("thresh3", thresh_img)

    return thresh_img

def find_arrow(base_img, arrow_img, base_gray, arrow_gray):
    """
    Both params are gray  value images.
    """

    # Compute diff
    # diff_img = _compute_diff(base_gray, arrow_gray)
    diff_img = _process_change(base_gray, arrow_gray, 70)

    # Separate the channels
    b,g,r = cv2.split(arrow_img)

    # Extract the blue
    diff = cv2.subtract(b, r, dtype=cv2.CV_16S)
    blue = np.greater(diff, 50) # TODO: Use static parameter like this?

    # De-normalize. range(0-255)
    blue = Utility.convert_to_cv(blue)

    # cv2.imshow("Blue", blue)

    # TODO: find out what is needed in the isolate_arrow method.

    # cv2.imshow("DIFF 1", diff_img)
    # Some morphology and thresholding to isolate arrow further
    isolated_arrow_img = _isolate_arrows(diff_img)
    cv2.imshow("isolated arrow", isolated_arrow_img)

    return isolated_arrow_img

    # Find coordinates within picture based on isolated arrows
    # coordinates = _locate_arrow(isolated_arrow_img)
    # print "Coordinates of ", coordinates

    # TODO: return coordinates here?


    # for c in coordinates:
    #     cv2.circle(arrow_img, c, 2, 255)
    # cv2.imshow("points", arrow_img)

def get_coordinate(arrow_img):
    return _locate_arrow(arrow_img)

def extract_arrow(arrow1, arrow2):
    # (not arrow1) and arrow2
    strel = np.ones((5, 5), np.uint8)
    arrow1 = cv2.morphologyEx(arrow1, cv2.MORPH_DILATE, strel)
    # cv2.imshow("not this", arrow1)
    # cv2.imshow("adn with this", arrow2)
    return cv2.bitwise_and(cv2.bitwise_not(arrow1), arrow2)
    # return cv2.bitwise_xor(arrow1, arrow2)

def join_contours(arrow_img):
    image, contours, hierarchy = cv2.findContours(arrow_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    arrow_img = cv2.drawContours(arrow_img, contours, -1, 255,-1)
    # cv2.drawContours(blurred,contours,-1,255,-1)
    cv2.imshow("after count", arrow_img)
    if len(contours) == 0:
        return None
    elif len(contours) == 1:
        return True

    # Find centroids
    centers = []
    for cnt in contours:
        # M = cv2.moments(cnt)
        #
        # if M['m00'] == 0.0 or M['m00'] == 0.0:
        #     return None
        #
        # centroid_x = int(M['m10']/M['m00'])
        # centroid_y = int(M['m01']/M['m00'])
        x, y = Utility.get_centroid(cnt)
        if x < 10 or y < 10:
            continue
        centers.append(Utility.get_centroid(cnt))

    # Draw line from each centroid to the next
    cnt = centers[0]
    for p in centers:
        cv2.line(arrow_img, cnt, p, (255,255,255), thickness=2)
    return True

def _isolate_arrows(diff_img):
    """
    Takes a grayscale img, computes diff and some morphology.
    Returns bw-img with arrow contours.
    """

    strel = np.ones((4, 4), np.uint8)
    open = cv2.morphologyEx(diff_img, cv2.MORPH_OPEN, strel)
    # cv2.imshow("open 2", open)

    strel2 = np.ones((20, 20), np.uint8)
    dilate = cv2.morphologyEx(open, cv2.MORPH_DILATE, strel2)
    # cv2.imshow("dialted 3", dilate)

    anded = cv2.bitwise_and(diff_img, dilate)
    # cv2.imshow("anded 4", anded)



    # blur = cv2.GaussianBlur(open, (5, 5), 0)
    # cv2.imshow("blur 3", blur)

    # ret, thresh_img = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # cv2.imshow("thresh 4", thresh_img)

    # strel2 = np.ones((5, 5), np.uint8)
    # open_thresh = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, strel2)
    # cv2.imshow("open 5", open_thresh)

    # Dilating to merge blobs belonging to the same arrow together
    # strel3 = np.ones((5, 5), np.uint8)
    # open_thresh = cv2.morphologyEx(open_thresh, cv2.MORPH_DILATE, strel3)

    return anded

def _locate_arrow(bw_img):
    """
    Takes a bw-img with arrow contours.
    Return coordinates of arrows.
    """

    # First merge arrows that are separated
    # cv2.imshow("bw", bw_img)


    image, contours, hierarchy = cv2.findContours(bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bw_img = cv2.drawContours(bw_img, contours, -1, 255)

    # cv2.imshow("cnt", bw_img)

    # (x,y),(MA,ma),angle = cv2.fitEllipse(contours[0])
    #
    # print x, y
    # print MA, ma
    # print angle

    # Find centroid of every cnt and point furthest away from it
    centers = []
    for cnt in contours:
        #
        # M = cv2.moments(cnt)
        # centroid_x = int(M['m10']/M['m00'])
        # centroid_y = int(M['m01']/M['m00'])
        centers.append(Utility.get_centroid(cnt))
        # cv2.circle(bw_img, (centroid_x, centroid_y), 2, 255)

    # find extreme points
    index = 0
    coordinates = []
    for cnt in contours:

        leftmost = cnt[cnt[:,:,0].argmin()][0]
        rightmost = cnt[cnt[:,:,0].argmax()][0]
        topmost = cnt[cnt[:,:,1].argmin()][0]
        bottommost = cnt[cnt[:,:,1].argmax()][0]
        centroid = centers[index]

        points = [leftmost, rightmost, topmost, bottommost]

        # Distance from centroid
        ld = np.linalg.norm(leftmost - centroid)
        rd = np.linalg.norm(rightmost - centroid)
        td = np.linalg.norm(topmost - centroid)
        bd = np.linalg.norm(bottommost - centroid)

        distances = [ld, rd, td, bd]
        max_index = np.argmax(distances)

        coordinates.append(tuple(points[max_index]))

        # cv2.circle(bw_img, tuple(points[max_index]), 2, 255)
        # cv2.circle(bw_img, rightmost, 2, 255)
        # cv2.circle(bw_img, topmost, 2, 255)
        # cv2.circle(bw_img, bottommost, 2, 255)

        # print leftmost, rightmost, topmost, bottommost
        # print centers[index]
        #
        # print type(leftmost), leftmost
        # print type(centers[index]), centers[index]
        # print type(lt), lt
        # print type(c), c

        # dist = np.linalg.norm(lt - c)

        # print dist

        index += 1

    print coordinates


    # cv2.imshow("mass", bw_img)

    # rows, cols = bw_img.shape[:2]
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

    return coordinates

def _compute_diff(base_img, new_img):
    # Compute diff
    diff_img = cv2.subtract(new_img, base_img, dtype=cv2.CV_16S)
    diff_img = cv2.convertScaleAbs(diff_img)
    return diff_img

# TODO: What change happened? # Arrow? --> isolate and locate. # Camchange --> return that.

# Testing
# TODO: findarrow works, now: classify change!!!

# base = cv2.imread("C:\Users\Torgeir\Desktop\\base.jpg")
# d1 = cv2.imread("C:\Users\Torgeir\Desktop\d1.jpg")
# d2 = cv2.imread("C:\Users\Torgeir\Desktop\d2.jpg")
#
# b = Board()
# # # center=tuple
# # #ellipse = ((center),(width,height of bounding rect), angle)
# center, ellipse, mask = b.detect(base)
# #
# # print("center", center)
# # print("centereclipse", ellipse[0])
# # print("width, height", ellipse[1])
# #
# grayb = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
# d1g = cv2.cvtColor(d1, cv2.COLOR_BGR2GRAY)
# d2g = cv2.cvtColor(d2, cv2.COLOR_BGR2GRAY)
# #
# width = len(base[0])
# height = len(base)
# nof_pixels = width * height
# max_change = nof_pixels * 255
#
# bounding_offset = 60
#
# x_offset = (ellipse[1][0] / 2)
# x_center = ellipse[0][0]
#
# y_offset = ellipse[1][1] / 2
# y_center = ellipse[0][1]
#
# minx = x_center - x_offset - bounding_offset
# maxx = x_center + x_offset + bounding_offset
# miny = y_center - y_offset - bounding_offset
# maxy = y_center + y_offset + bounding_offset
#
# percent, change = classify_change(grayb, d1g, 0.01, max_change, (int(minx, miny)), (maxx, maxy))
#

"""

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
"""
# cv2.waitKey(0)
# cv2.destroyAllWindows()
