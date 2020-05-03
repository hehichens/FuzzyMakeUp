'''
Organ
'''

import dlib
import cv2
import numpy as np
from Constant import *
from FuzzyThreshold import FT

class Organ():
    def __init__(self, img, landmark, name):
        self.name = name
        self.img = img
        self.img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.Timg, self.Timg_hsv = self.img, self.img_hsv
        self.landmark = landmark
        self.left, self.top, self.right, self.bottom = self.get_rect()
        self.Oimg =  self.get_patch(self.img) # top:bottom, left:right
        self.Oimg_hsv = self.get_patch(self.img_hsv)

        # cv2.imshow(self.name, self.Oimg)
        # cv2.rectangle(self.img, (self.left, self.top), (self.right, self.bottom), RED) # draw the organ with rectangle
        self.mask = self.get_mask()
        # self.Ksize = self.get_ksize()

    def get_rect(self):
        ys, xs = self.landmark[:, 1], self.landmark[:, 0]
        top, bottom, left, right = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
        self.size = (bottom - top) * (right - left)
        move = int(np.sqrt(abs(self.size) // MOVE_RATE))
        return left, top, right, bottom
        # return left - move, top - move, right + move, bottom + move

    def get_patch(self, img):

        return img[max(self.top, 0):min(self.bottom, self.img.shape[1]),
                        max(self.left, 0):min(self.right, self.img.shape[0])]

    def get_mask(self):
        landmark_re = self.landmark.copy()
        landmark_re[:, 1] -= np.max([self.top, 0])
        landmark_re[:, 0] -= np.max([self.left, 0])
        Points = cv2.convexHull(landmark_re)

        mask = np.zeros(self.Oimg.shape[:2], dtype=np.float64)
        cv2.fillConvexPoly(mask, Points, 1)
        mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
        # cv2.imshow(self.name, mask)
        mask = (mask > 0) * 1.0
        return mask

    def confirm(self):
        self.img[:], self.img_hsv[:] = self.Timg[:], self.Timg_hsv[:]

    def update_img(self):
        self.Timg[:], self.Timg_hsv[:] = self.img[:], self.img_hsv[:]

    def Brightening(self, rate=GOLD_RATE):
        '''
        make the image color more vivid
        '''
        self.confirm()
        self.Oimg_hsv[:, :, 1] = np.minimum(self.Oimg_hsv[:, :, 1] + self.Oimg_hsv[:, :, 1] * self.mask[:, :, 1] * rate, 255)
        self.Oimg_hsv[:, :, 1] = cv2.GaussianBlur(self.Oimg_hsv[:, :, 1], BLUR_SIZE, 0)
        self.img[:] = cv2.cvtColor(self.img_hsv, cv2.COLOR_HSV2BGR)[:]  # update image
        self.update_img()
        # cv2.rectangle(self.img, (self.left, self.top), (self.right, self.bottom), (255, 0, 0))
        # cv2.imshow("change", cv2.cvtColor(self.Oimg_hsv, cv2.COLOR_HSV2BGR))

    def Whitening(self, rate=0.15):
        '''
        make the image color more white
        '''
        self.confirm()
        gray_img = cv2.cvtColor(self.Oimg, cv2.COLOR_BGR2GRAY)
        mask = FT(gray_img)
        self.Oimg_hsv[:, :, -1] = np.minimum(self.Oimg_hsv[:, :, -1] + self.Oimg_hsv[:, :, -1] * mask * rate, 255)
        self.Oimg_hsv[:, :, -1] = cv2.GaussianBlur(self.Oimg_hsv[:, :, -1], BLUR_SIZE, 0)
        self.img[:] = cv2.cvtColor(self.img_hsv, cv2.COLOR_HSV2BGR)[:]  # update image
        self.update_img()
        # cv2.imshow("temp_img", self.img)

    def Smooth(self):
        self.confirm()
        self.img_hsv[:] = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)[:]
        self.update_img()

class Face(Organ):
    def __init__(self, img, face_landmark, name):
        super(Face, self).__init__(img, face_landmark, name)


    def get_rect(self):
        ys, xs = self.landmark[:, 1], self.landmark[:, 0]
        top, bottom, left, right = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
        self.size = (bottom - top) * (right - left)
        move = int(np.sqrt(abs(self.size) // MOVE_RATE))
        return left, top - 2*move, right, bottom