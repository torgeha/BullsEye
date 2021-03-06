
import cv2
import time
import threading
from Queue import Queue

from cv.framediff import classify_change, find_arrow, extract_arrow, get_coordinate, join_contours
from cv.board import Board
from dartgame.classical import Classical

print cv2.__version__

class Camera:

    def __init__(self, cam_interface, fps_limit, fps_eval, should_display, vid_src_path=None):
        # fps_limit --> the fps the feed should read frames
        # fps_eval --> the fps frames should be evaluated

        # self.buffer = CameraBuffer(cam_interface, vid_src_path)
        self.fps_limit = fps_limit
        self.fps_eval = fps_eval
        self.fps_fraction = self.fps_limit / self.fps_eval

        self.video = cv2.VideoCapture(cam_interface)
        self.is_live = True
        if vid_src_path:
            self.is_live = False
            self.video = cv2.VideoCapture(vid_src_path)

        self.should_display = should_display

        ret, frame = self.video.read()
        self.width = len(frame[0])
        self.height = len(frame)
        self.board = Board()
        self.game = Classical()

        self.capture()


    def capture(self):

        ret, base_frame = self.video.read()
        # base_frame = self.buffer.get_frame()
        last_frame = base_frame
        roi_last = None
        bounding_box = None
        base_frame_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
        last_frame_gray = base_frame_gray

        # All changes under this will be ignored
        percent_threshold = 0.05

        # Fps stuff
        loop_delta = 1./self.fps_limit
        current_time = target_time = time.clock()
        last_frame_changed = False
        last_frame_arrow = False

        # If live feed, dont wait, else wait
        wait_per_frame = 25
        if self.is_live:
            wait_per_frame = 1

        # Padding for the board bounding box
        bounding_offset = 70

        # skip_frame = False

        nof_arrows = 0
        arrow_img1 = None
        arrow_img2 = None

        while (True):
            # Sleep management. Limit fps.
            if self.is_live: # only sleep
                target_time += loop_delta
                sleep_time = target_time - time.clock()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            # Loop frequency evaluation, prints actual fps
            previous_time, current_time = current_time, time.clock()
            time_delta = current_time - previous_time
            print 'frequency: %s' % (1. / time_delta)

            ret, new_frame = self.video.read()
            # new_frame = self.buffer.get_frame()
            if new_frame == None:
                print "END OF VIDEO, BREAKING"
                break # no more frames to read
            cv2.imshow("FEED", new_frame)
            cv2.waitKey(1)

            # Detect board and use it to base the finding of changes
            # Bounding box is none before a baseframe is found
            should_set_baseframe = False
            if bounding_box is None:
                # Try to find board if the boundingbox is not set
                bounding_box = self._get_bounding_box(new_frame, bounding_offset)

                if bounding_box is None:
                    continue

                should_set_baseframe = True

            # Code will never reach here if boundingbox is not set
            # Convert current frame to gray
            new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

            # This happens first time ONLY
            if should_set_baseframe:
                base_frame = new_frame
                base_frame_gray = new_frame_gray
                last_frame = base_frame
                last_frame_gray = base_frame_gray
                last_frame_changed = False
                print "************* NEW BASE SET ****************"
                print " ..and bounding box is", bounding_box
                cv2.imshow("BASE", base_frame)
                # skip_frame = True
                continue

            # Crop frame, find Roi of current frame and last frame
            roi_new = new_frame_gray[bounding_box[0][1]:bounding_box[1][1], bounding_box[0][0]:bounding_box[1][0]]
            roi_last = last_frame_gray[bounding_box[0][1]:bounding_box[1][1], bounding_box[0][0]:bounding_box[1][0]]

            # Find max change based on ROI
            max_change = self._get_max_change(roi_new)

            # Find change between this frame and the previous frame
            change_percent, change = classify_change(roi_last,
                                                     roi_new,
                                                     percent_threshold,
                                                     max_change)
            print "change: ", change_percent, " value: ", change

            if cv2.waitKey(wait_per_frame) & 0xFF == ord('q'):
                break

            if change == 0: # Nothing changed, continue
                if last_frame_changed:
                    # last frame changed and is now stable, set new base_frame
                    # By setting the boudning_box to None, the next loop will set new base when bounding box is found
                    bounding_box = None

                elif last_frame_arrow:
                    # Arrow detected in last frame, now stable, find location
                    # TODO since there was no change from last frame, find arrow from base_frame and arrow_frame
                    frame_with_arrow = last_frame
                    # cv2.imshow("arrow", frame_with_arrow)
                    # cv2.imshow("arrowgray", last_frame_gray)

                    coordinate = None

                    if nof_arrows == 0:
                        arrow_img1 = find_arrow(base_frame, last_frame, base_frame_gray, last_frame_gray)
                        joined_cunt = join_contours(arrow_img1)
                        cv2.imshow("joined 1", arrow_img1)
                        if not (joined_cunt == None):
                            print "one arrow"
                            coordinate = get_coordinate(arrow_img1.copy())
                            nof_arrows += 1
                            coordinate = coordinate[0]
                            score = self.get_score(coordinate)
                            print "Score: ", score
                            self.game.add_hit(score, coordinate[0], coordinate[1])
                    elif nof_arrows == 1:
                        arrow_img2 = find_arrow(base_frame, last_frame, base_frame_gray, last_frame_gray)
                        arr = extract_arrow(arrow_img1, arrow_img2)
                        # cv2.imshow("Extracted", arr)
                        joined_cunt = join_contours(arr)
                        cv2.imshow("joined 2", arr)
                        if not (joined_cunt == None):
                            print "two arrow"
                            coordinate = get_coordinate(arr)
                            nof_arrows += 1
                            coordinate = coordinate[0]
                            score = self.get_score(coordinate)
                            print "Score: ", score
                            self.game.add_hit(score, coordinate[0], coordinate[1])
                    elif nof_arrows == 2:
                        arrow_img3 = find_arrow(base_frame, last_frame, base_frame_gray, last_frame_gray)
                        arr = extract_arrow(arrow_img2, arrow_img3)
                        joined_cunt = join_contours(arr)
                        if not (joined_cunt == None):
                            print "Third Arrow"
                            cv2.imshow("joined 3", arr)
                            coordinate = get_coordinate(arr)
                            nof_arrows = 0
                            coordinate = coordinate[0]
                            score = self.get_score(coordinate)
                            print "Score: ", score
                            self.game.add_hit(score, coordinate[0], coordinate[1])
                            self.game.next_player()


                    print "Coordinates", coordinate

                    last_frame_arrow = False
                continue
            elif change == 1: # New arrow, wait until stable
                last_frame_arrow = True
                arrow_frame = new_frame

            elif change == 2: # Camera change, check next until no change, then set base
                last_frame_changed = True

            # roi_last = roi_new
            last_frame = new_frame
            last_frame_gray = new_frame_gray

        cv2.destroyAllWindows()

    def _find_base_frame(self):
        """
        Return baseframe
        """

    def get_score(self, coordinate):
        # cv2.imshow("gettingscorefromthis", self.point_mask)
        # print "Getting this", coordinate, "x is", coordinate[0], " y is ", coordinate[1]
        return self.point_mask[coordinate[1]][coordinate[0]]

    def _get_max_change(self, roi):
        width = len(roi[0])
        height = len(roi)
        nof_pixels = width * height
        return nof_pixels * 255

    def _get_bounding_box(self, frame, bounding_offset):
        """
        Will not return until board is detected. Returns bounding box of board
        """

        # Try to find board if the boundingbox is not set
        center, ellipse, mask = self.board.detect(frame)

        # Should not be None
        if center is None:
            print("skipping frame")
            return None
        if ellipse is None:
            print("skipping frame")
            return None
        if mask is None:
            print("skipping frame")
            return None

        self.point_mask = mask
        # cv2.imshow("mask", mask)

        x_offset = (ellipse[1][0] / 2)
        x_center = ellipse[0][0]

        y_offset = ellipse[1][1] / 2
        y_center = ellipse[0][1]

        minx = max(0, x_center - x_offset - bounding_offset)
        maxx = min(self.width, x_center + x_offset + bounding_offset)
        miny = max(0, y_center - y_offset - bounding_offset)
        maxy = min(self.height, y_center + y_offset + bounding_offset)
        return ((int(minx), int(miny)), (int(maxx), int(maxy)))

class CameraBuffer:
    # DO NOT USE!!!

    def __init__(self, interface, path=None):
        """
        Interface --> live video
        path --> path to file
        """
        self.live = False
        if path == None:
            self.live = True
        self.video = cv2.VideoCapture(interface)
        if path:
            self.video = cv2.VideoCapture(path)

        self.buffer = Queue(1000)
        if not self.live:
            t = threading.Thread(target=self._capture)
            t.daemon = True
            t.start()

    def _capture(self):
        ret, frame = self.video.read()
        while ret:
            self.buffer.put(frame)
            ret, frame = self.video.read()
        self.video.release()

    def get_frame(self, delay=None):
        if self.live:
            ret, frame = self.video.read()
            if ret:
                return frame
            return None
        if not self.buffer.empty():
            return self.buffer.get()
        return None

if __name__ == "__main__":
    cam = Camera(2, 5, 5, True)


