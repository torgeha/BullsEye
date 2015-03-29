
import cv2
import numpy as np

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

    thresh = diff_frame(base_frame, new_frame, 20)

    # Sum of pixels
    pixel_sum = cv2.sumElems(thresh)[0] # TODO: more efficient way to do this?

    # Percent of frame that differs
    change_percent = (pixel_sum / max_change) * 100

    # TODO: tune these parameters to dart board
    if change_percent < percent_threshold: # Change is under percent_threshold, nothing changed
        return (change_percent, 0)
    elif change_percent < 1.5: # Change is more than thresh, less than 5 --> arrow
        return (change_percent, 1)
    else: # More has changed --> camera
        return (change_percent, 2)

def diff_frame(base_img, new_img, thresh):
    """
    Takes two grayvalue images. Computes diff, blurs and global threshold. Returns bw image.
    """

    # Use int16 arrays for negative values
    # new_img = np.array(new_img, dtype="int16")
    # base_img = np.array(base_img, dtype="int16")

    cv2.imshow("new", new_img)
    cv2.imshow("base", base_img)

    # Compute diff
    diff_img = cv2.subtract(new_img, base_img, dtype=cv2.CV_16S)
    diff_img = cv2.convertScaleAbs(diff_img)

    print np.amin(diff_img)
    # diff_img = diff_img + 100
    cv2.imshow("diff", diff_img)

    # Gaussian blur
    blur = cv2.GaussianBlur(diff_img, (5, 5), 0)
    cv2.imshow("blur", blur)

    # Global threshold
    ret, thresh_img = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY) # TODO: thresh value as parameter?
    return thresh_img


def locate_arrow(base_frame, dart_frame):
    """
    Return coordinates of arrow point.
    """
    # compute diff, and whatever else is needed to find location of arrow on the board
    pass


# Testing

base = cv2.imread("C:\Users\Torgeir\Desktop\\base.jpg")
d1 = cv2.imread("C:\Users\Torgeir\Desktop\d1.jpg")
d2 = cv2.imread("C:\Users\Torgeir\Desktop\d2.jpg")

grayb = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
d1g = cv2.cvtColor(d1, cv2.COLOR_BGR2GRAY)
d2g = cv2.cvtColor(d2, cv2.COLOR_BGR2GRAY)

# cv2.imshow("base", grayb)
# cv2.imshow("d1", d1g)
# cv2.imshow("d2", d2g)
diff1 = diff_frame(grayb, d2g, 50)

cv2.imshow("diff1", diff1)


cv2.waitKey(0)
cv2.destroyAllWindows()
