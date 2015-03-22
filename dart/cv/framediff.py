
import cv2

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


def what_changed(base_frame, new_frame, percent_threshold, max_change):
    """
    Return 0: Nothing
    Return 1: New arrow
    Return 2: Camera change

    Returns tuple of (percent, value)
    """

    # TODO: expand to use detectors based on dart board location

    # Compute diff
    diff_frame = cv2.subtract(new_frame, base_frame)

    # Gaussian blur
    blur = cv2.GaussianBlur(diff_frame, (5, 5), 0)

    # Global threshold
    ret, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) # TODO: thresh value as parameter?

    # Sum of pixels
    pixel_sum = cv2.sumElems(thresh)[0] # TODO: more efficient way to do this?

    # Percent of frame that differs
    change_percent = (pixel_sum / max_change) * 100

    # TODO: tune these parameters to dart board
    if change_percent < percent_threshold: # Change is under percent_threshold, nothing changed
        return (change_percent, 0)
    elif change_percent < 5.0: # Change is more than thresh, less than 5 --> arrow
        return (change_percent, 1)
    else: # More has changed --> camera
        return (change_percent, 2)


