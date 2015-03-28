
import cv2
import time

from framediff import classify_change

print cv2.__version__

class Camera:

    def __init__(self, cam_interface, fps_limit, fps_eval, should_display, vid_src_path=None):
        # fps_limit --> the fps the feed should read frames
        # fps_eval --> the fps frames should be evaluated

        self.fps_limit = fps_limit
        self.fps_eval = fps_eval
        self.fps_fraction = self.fps_limit / self.fps_eval

        self.cap = cv2.VideoCapture(cam_interface)

        if vid_src_path:
            self.cap = cv2.VideoCapture(vid_src_path)

        self.should_display = should_display

        ret, frame = self.cap.read()
        self.width = len(frame[0])
        self.height = len(frame)
        self.nof_pixels = self.width * self.height
        self.max_change = self.nof_pixels * 255

        print self.max_change

        self.capture()


    def capture(self):

        ret, base_frame = self.cap.read()
        last_frame = base_frame
        base_frame_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
        last_frame_gray = base_frame_gray

        percent_threshold = 1.0 # Percent of the frame that is allowed to change without consequences

        # Fps stuff
        loop_delta = 1./self.fps_limit
        current_time = target_time = time.clock()
        last_frame_changed = False
        last_frame_arrow = False

        # Evaluation fps stuff
        loop_count = 0

        while (True):
            # Sleep management. Limit fps.
            target_time += loop_delta
            sleep_time = target_time - time.clock()
            if sleep_time > 0:
                time.sleep(sleep_time)
                pass

            # Loop frequency evaluation, prints actual fps
            # previous_time, current_time = current_time, time.clock()
            # time_delta = current_time - previous_time
            # print 'frequency: %s' % (1. / time_delta)

            ret, new_frame = self.cap.read()
            if not ret:
                break # no more frames to read

            # Continue if frame should not be evaluated
            loop_count += 1
            if loop_count <= self.fps_fraction:
                continue
            loop_count = 0

            print "lfc", last_frame_changed
            print "lfa", last_frame_arrow

            # Display video feed?
            if self.should_display:
                cv2.imshow('Feed', new_frame)

            new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

            # TODO: nothing changed? --> continue
            # TODO: camera/arrow --> wait x frames and find arrow_pos/set new base frame

            change_percent, change = classify_change(last_frame_gray, new_frame_gray, percent_threshold, self.max_change)

            print "change: ", change_percent, " value: ", change

            if change == 0: # Nothing changed, continue
                if last_frame_changed:
                    # last frame changed and is now stable, set new base_frame
                    base_frame = new_frame
                    base_frame_gray = new_frame_gray
                    cv2.imshow('baseframe', base_frame)
                    last_frame_changed = False
                    print "**************************  NEW BASE"
                elif last_frame_arrow:
                    # Arrow detected in last frame, now stable, find location
                    # TODO since there was no change from last frame, find arrow from base_frame and arrow_frame
                    # Set last_frame_arrow = False
                    last_frame_arrow = False
                continue
            elif change == 1: # New arrow, wait until stable
                last_frame_arrow = True
                arrow_frame = new_frame

            elif change == 2: # Camera change, check next until no change, then set base
                last_frame_changed = True

            last_frame = new_frame
            last_frame_gray = new_frame_gray

            # Quit if "q" is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# Testing

path = "C:\Users\Torgeir\Desktop\dartH264"

path = path + "\dart-anglechange.mp4"
print path

cam = Camera(0, 25, 5, True, path)
