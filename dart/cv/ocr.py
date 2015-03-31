import cv2
import numpy as np
import scipy
from utility import Utility
from board import Board
import math
import sys
print(np.__version__)
from sklearn import neighbors
class DartLearner:

    def __init__(self, samples=None, responses=None):
        self.number = 20
        self.model = neighbors.KNeighborsClassifier(self.number, weights='distance')
        if samples and responses:
            self.train(samples, responses)

    def train(self, samples, targets):
        self.model.fit(samples, targets)
        return self.model

    def init_trained_model(self, data):
        #TODO: Should be a way to pre train the model for to avoid online learning
        pass

    def test(self, image, ellipse):
        #Let model classify training example
        mask = DartHelper.create_mask(image, ellipse)
        img,contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image, contours, -1, (0,255,255), thickness=-1)
        cv2.imshow("contours", image)
        groups = DartHelper.group_numbers(contours, ellipse)

        for n in groups:
            avg_area, points, rect = DartHelper.get_group_description(n)
            if avg_area >20:
                [x,y,w,h] = rect
                #print("group", x,y,w,h)
                if  h>4:
                    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
                    roi = DartHelper.create_roi(mask, [x,y,w,h])
                    roi = DartHelper.reshape_roi(roi)
                    result = self.model.predict(roi)
                    string = str(int((result)))
                    print(string)
                    cv2.putText(image,string,(x,y+h),0,1,(20,255,20), thickness=1)
        cv2.imshow("Result", image)
        cv2.waitKey(-1)


class DartHelper:
    WIDE_FACTOR = 1.25


    @staticmethod
    def create_mask(image, ellipse):
        blurred = cv2.GaussianBlur(image,(5,5),0)
        grey = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        ret,thresh = cv2.threshold(grey,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        number_mask = thresh
        number_mask = cv2.ellipse(number_mask, ellipse, (0,0,0), thickness=-1)
        wide_ellipse = Utility.scale_ellipse(ellipse, DartHelper.WIDE_FACTOR)
        mask = np.zeros(number_mask.shape, np.uint8)

        cv2.ellipse(mask, wide_ellipse, 1, thickness=-1)
        number_mask = cv2.multiply(number_mask, mask)
        #number_mask = cv2.erode(number_mask,  np.ones((2,2),np.uint8))
        return number_mask

    @staticmethod
    def get_group_description(group):
         points = np.vstack(c[2] for c in group)
         x,y,w,h = rect= cv2.boundingRect(points)
         return w*h, points, rect

    @staticmethod
    def group_numbers(contours,ellipse, max_factor=2.9):
        #Prune away to big contours, like outline caused by not-fitted ellipse
        max_w , max_h = ellipse[1][0]*0.09,ellipse[1][1]*0.09
        small_and_big_contours = []
        for i,cnt in enumerate(contours):
            [x,y,w,h] = cv2.boundingRect(cnt)
            if max_w<w or max_h<h or (w <2 or h<2):
                small_and_big_contours.append(i)
        contours = np.delete(contours, small_and_big_contours)
        groups = [] #20 groups ideally
        avg_area = 0
        for cnt in contours:
            avg_area += cv2.contourArea(cnt)
        avg_area= avg_area / len(contours) #TODO: or 20?
        max_dist = np.sqrt(avg_area) * max_factor
        if len(contours)>1:
            x,y = Utility.get_centroid(contours[0])
            groups.append([(x,y, contours[0])])
            for i in range(1, len(contours)):
                cnt = contours[i]
                x,y = Utility.get_centroid(cnt)
                found = False
                for g in groups:
                    ax, ay = DartHelper.get_avg_group(g)
                    dist = math.sqrt((ax-x)**2 + (ay-y)**2)
                    if dist < max_dist:
                        found = True
                        g.append((x,y,cnt))
                        break
                if not found:
                    groups.append([(x,y, cnt)])
            return groups
        return []

    @staticmethod
    def create_roi(mask, rectangle, width=20, height=20):
        x,y,w,h = rectangle
        roi = mask[y:y+h,x:x+w]
        roi = cv2.resize(roi,(width,height))
        return roi

    @staticmethod
    def reshape_roi(roi, length=400):
        return np.float32(roi.reshape((1,length)))

    @staticmethod
    def get_avg_group(group):
        sx = sum(x[0] for x in group)
        sy = sum(y[1] for y in group)
        return sx/len(group), sy/len(group)


class DartTrainingDataCreator:
    SAMPLE_FILENAME = "generalsamples.data"
    RESPONSE_FILENAME = "generalresponses.data"
    ENTER = 13
    ESCAPE = 27

    def sample(self, image, ellipse, sample_file=SAMPLE_FILENAME, response_file=RESPONSE_FILENAME):
        mask = DartHelper.create_mask(image, ellipse)
        img,contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        groups = DartHelper.group_numbers(contours, ellipse)
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






train = True

t = "C:\Users\Olav\OneDrive for Business\BullsEye\Pictures\dartboard23.png"
img = cv2.imread(t, 1)
b = Board()
ellipse = b.detect_ellipse(img)
if train:
    learner = DartTrainingDataCreator()
    samples, responses = learner.sample(img, ellipse)
else:
    samples = np.loadtxt('generalsamples.data',np.float32)
    responses = np.loadtxt('generalresponses.data',np.float32)
    responses = responses.reshape((responses.size,1))
    learner = DartLearner()
    learner.train(samples, responses)
    learner.test(img, ellipse) #TODO: img
