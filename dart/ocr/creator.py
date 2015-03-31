from ocr.learner import DartHelper, DartLearner
import cv2
from cv.board import Board
import numpy as np


class DartTrainingDataCreator:
    SAMPLE_FILENAME = "dart_samples.data"
    RESPONSE_FILENAME = "dart_targets.data"
    ENTER = 13
    ESCAPE = 27

    def sample(self, image, ellipse, sample_file=SAMPLE_FILENAME, response_file=RESPONSE_FILENAME):
        contours, mask, groups = DartHelper.create_number_descriptions(image, ellipse)
        samples, responses = self.get_numbers(groups, mask, image)
        self.append_to_file(samples, sample_file)
        self.append_to_file(responses, response_file)
        return samples, responses

    def append_to_file(self, data, filename):
        file_handle = file(filename , 'a')
        np.savetxt(file_handle,data)
        file_handle.close()

    def request_target_from_user(self, image, roi):
        [x,y,w,h] = roi
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('Press enter when finished',image)
        entered_keys = []
        while True:
            key = cv2.waitKey(0)
            if  key is DartTrainingDataCreator.ENTER:
                break
            entered_keys.append(key)
        return entered_keys

    def get_numbers(self, groups, mask, image):
        samples =  np.empty((0,400))
        responses = []
        for n in groups:
            avg_area, points, rect =  DartHelper.get_group_description(n)
            if avg_area >20:
                [x,y,w,h]= rect
                if  h>4:
                    keys = [i for i in range(48,58)]
                    entered_keys = self.request_target_from_user(image, rect)

                    if len(entered_keys)>0 and (k in keys for k in entered_keys):
                        n = int(''.join(map(chr, entered_keys)))
                        responses.append(n)
                        roi = DartHelper.create_roi(mask, rect)
                        sample = DartHelper.reshape_roi(roi)
                        samples = np.append(samples,sample,0)
                        print(responses)
        responses = np.array(responses,np.float32)
        responses = responses.reshape((responses.size,1))
        return samples, responses






train = False

t = "C:\Users\Olav\OneDrive for Business\BullsEye\Pictures\dart2.png"
img = cv2.imread(t, 1)
b = Board()
ellipse = b.detect_ellipse(img)
if train:
    learner = DartTrainingDataCreator()
    samples, responses = learner.sample(img, ellipse)
else:
    samples = np.loadtxt("dart_samples.data",np.float32)
    responses = np.loadtxt("dart_targets.data",np.float32)
    responses = responses.reshape((responses.size,1))
    learner = DartLearner()
    learner.train(samples, responses)
    learner.test(img, ellipse) #TODO: img
