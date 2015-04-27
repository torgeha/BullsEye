import cv2
import numpy as np
from cv.utility import Utility
import math
from sklearn import neighbors


class DartLearner:

    SAMPLE_FILENAME = "ocr/dart_samples.data"
    RESPONSE_FILENAME = "ocr/dart_targets.data"

    def __init__(self, samples=None, responses=None):
        self.number = 20
        self.model = neighbors.KNeighborsClassifier(self.number, weights='distance')
        if samples and responses:
            if isinstance(samples, basestring):
                # print(samples)
                samples = np.loadtxt(samples,np.float32)
            if isinstance(responses, basestring):
                responses = np.loadtxt(responses,np.float32)
            self.train(samples, responses)

    def train(self, samples, targets):
        self.model.fit(samples, targets)
        return self.model

    def init_trained_model(self, data):
        #TODO: Should be a way to pre train the model for to avoid online learning
        pass

    def classify(self, roi):
        return self.model.predict(roi)

    def classify_all(self, mask, groups):
        classifications = []
        for n in groups:
            avg_area, points, rect = DartHelper.get_group_description(n)
            [x,y,w,h] = rect
            if avg_area >20 and h>4:
                roi = DartHelper.create_roi(mask, [x,y,w,h])
                roi = DartHelper.reshape_roi(roi)
                result = int((self.model.predict(roi)))

                classifications.append((rect, result))
            else:
                classifications.append((rect, -1))
        #TODO: Mend errors. Take advantage of knowing order and that a prediction should only apply once
        return classifications


    def test(self, image, ellipse):
        #Let model classify training example
        contours, mask, groups = DartHelper.create_number_descriptions(image, ellipse)
        predictions = self.classify_all(mask, groups)
        for i,n in enumerate(groups):
            avg_area, points, rect = DartHelper.get_group_description(n)
            [x,y,w,h] = rect
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
            cv2.putText(image,str(predictions[i][1]),(x,y+h),0,1,(20,255,20), thickness=1)
        cv2.imshow("Result", image)
        cv2.waitKey(-1)


class DartHelper:
    WIDE_FACTOR = 1.25

    @staticmethod
    def create_number_descriptions(image, ellipse):
        mask = DartHelper.create_mask(image, ellipse)
        img,contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image, contours, -1, (0,255,255), thickness=-1)
        groups = DartHelper.group_numbers(contours, ellipse)
        return contours, mask, groups

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
        #TODO: Resize keep aspect ratio. Scale down to 20 for longest axis and use padding
        #Might reduce accuracy. Use caution. Maybe introduce another feature to differenciate?
        print(DartHelper.get_size(w,h, width, height))
        s = DartHelper.get_size(w,h, width, height)
        roi = cv2.resize(roi,s[0])
        roi = cv2.copyMakeBorder(roi,s[1][1], s[1][1], s[1][0], s[1][0], cv2.BORDER_CONSTANT, 0)
        return roi

    @staticmethod
    def get_size(w,h, nw, nh):
        if w>h:
            sh = int(math.floor((float(h)/w)*nw))
            sh = sh + sh%2
            return ((nw, sh),(0,(nh-sh)/2))
        elif h>w:
            sw = int(math.floor((float(w)/h)*nh))
            sw = sw + sw%2
            return ((sw, nh), ((nw-sw)/2 ,0))
        return ((nw, nh), (0, 0))

    @staticmethod
    def reshape_roi(roi, length=400):
        return np.float32(roi.reshape((1,length)))

    @staticmethod
    def get_avg_group(group):
        sx = sum(x[0] for x in group)
        sy = sum(y[1] for y in group)
        return sx/len(group), sy/len(group)


