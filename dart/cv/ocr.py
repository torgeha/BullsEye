import cv2
import numpy as np
from utility import Utility
from board import Board
import math
import sys

class DartLearner:

    def __init__(self, samples=None, responses=None):
        #self.model = cv2.KNearest() #TODO: cv2 Does not include ml in 3.0 yet
        if samples and responses:
            self.train(samples, responses)

    def train(self, samples, responses):
        self.model.train(samples, responses) #TODO: Change
        return self.model

    def init_trained_model(self, data):
        #TODO: Should be a way to pre train the model for to avoid online learning
        pass

    def test(self, image):
        #Let model classify training example
        mask = DartLearner.create_mask(image, ellipse)
        img,contours,hierarchy = cv2.findContours(mask,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        groups = DartHelper.group_numbers(contours)
        for n in groups:
            avg_area = 0
            a = np.vstack(c[2] for c in n)
            avg_area =  [cv2.contourArea(c) for c in a]
            if avg_area >20:
                s = np.vstack(x[2] for x in n)
                [x,y,w,h] = cv2.boundingRect(s)
                if  h>10:
                    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                    roi = mask[y:y+h,x:x+w]
                    roismall = cv2.resize(roi,(10,10))
                    roismall = roismall.reshape((1,100))
                    roismall = np.float32(roismall)
                    retval, results, neigh_resp, dists = self.model.find_nearest(roismall, k = 1)
                    string = str(int((results[0][0])))
                    cv2.putText(image,string,(x,y+h),0,1,(0,255,0))
class DartHelper:
    WIDE_FACTOR = 1.25

    @staticmethod
    def create_mask(image, ellipse):
        blurred = cv2.GaussianBlur(image,(5,5),0)
        grey = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        ret,thresh = cv2.threshold(grey,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        number_mask = thresh
        number_mask = cv2.ellipse(number_mask, ellipse, (0,0,0), thickness=-1)
        wide_ellipse = Utility.scale_ellipse(ellipse, 1.25)
        mask = np.zeros(number_mask.shape, np.uint8)

        cv2.ellipse(mask, wide_ellipse, 1, thickness=-1)
        number_mask = cv2.multiply(number_mask, mask)
        #number_mask = cv2.erode(number_mask,  np.ones((2,2),np.uint8))
        return number_mask

    @staticmethod
    def group_numbers(contours):
        groups = [] #20 groups ideally
        avg_area = 0
        for cnt in contours:
            avg_area += cv2.contourArea(cnt)
        avg_area / len(contours) #TODO: or 20?
        max_dist = np.sqrt(avg_area) * .5
        if len(contours)>1:
            x,y = Utility.get_centroid(contours[0])
            groups.append([(x,y, contours[0])])
            for i in range(1, len(contours)):
                cnt = contours[i]
                x,y = Utility.get_centroid(cnt)
                for g in groups:
                    ax, ay = DartHelper.get_avg_group(g)
                    dist = math.sqrt((ax-x)**2 + (ay-y)**2)
                    found = False
                    if dist < max_dist:
                        found = True
                        g.append((x,y,cnt))
                if not found:
                    groups.append([(x,y, cnt)])
            return groups

    @staticmethod
    def get_avg_group(group):
        sx = sum(x[0] for x in group)
        sy = sum(y[1] for y in group)
        return sx/len(group), sy/len(group)


class DartTrainingDataCreator:
    SAMPLE_FILENAME = "generalsamples.data"
    RESPONSE_FILENAME = "generalresponses.data"

    def sample(self, image, ellipse, sample_file=SAMPLE_FILENAME, response_file=RESPONSE_FILENAME):
        mask = DartHelper.create_mask(image, ellipse)
        img,contours,hierarchy = cv2.findContours(mask,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        groups = DartHelper.group_numbers(contours)
        samples, responses = self.get_numbers(groups, mask, image)
        self.append_to_file(samples, sample_file)
        self.append_to_file(responses, response_file)
        return samples, responses

    def append_to_file(self, data, filename):
        file_handle = file(filename , 'a')
        np.savetxt(file_handle,data)
        file_handle.close()

    def get_numbers(self, groups, mask, image):
        samples =  np.empty((0,100))
        responses = []
        for n in groups:
            avg_area = 0
            keys = [i for i in range(48,58)]
            a = np.vstack(c[2] for c in n)
            avg_area =  [cv2.contourArea(c) for c in a]
            if avg_area >20:
                s = np.vstack(x[2] for x in n)
                [x,y,w,h] = cv2.boundingRect(s)
                if  h>10:
                    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
                    roi = mask[y:y+h,x:x+w]
                    roismall = cv2.resize(roi,(10,10))
                    cv2.imshow('norm',image)
                    key = cv2.waitKey(0)
                    key2 = cv2.waitKey(0)
                if key == 27:  # (escape to quit)
                    sys.exit()
                elif key in keys and key2 in keys:
                    responses.append(int(str(chr(key) + chr(key2))))
                    sample = roismall.reshape((1,100))
                    samples = np.append(samples,sample,0)
                    print(responses)
        responses = np.array(responses,np.float32)
        responses = responses.reshape((responses.size,1))
        return samples, responses






train = True

if train:
    learner = DartTrainingDataCreator()
    b = Board()
    t = "C:\Users\Olav\OneDrive for Business\BullsEye\Pictures\dartboard.png"
    img = cv2.imread(t, 1)
    ellipse = b.detect_ellipse(img)
    samples, responses = learner.sample(img, ellipse)
else:
    samples = np.loadtxt('generalsamples.data',np.float32)
    responses = np.loadtxt('generalresponses.data',np.float32)
    responses = responses.reshape((responses.size,1))
    learner = DartLearner()
    learner.train(samples, responses)
    learner.test() #TODO: img
