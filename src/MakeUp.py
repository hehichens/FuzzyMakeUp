'''
detect the face and split the organs
'''

import dlib
import cv2
import numpy as np
from Constant import *
from Organ import *
from Face import *

# load the model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(LandmarkPath)


class MakeUp():
    def __init__(self, img):

        self.img = img
        self.organs_name=['jaw', 'mouth', 'nose', 'left eye', 'right eye', 'left brow', 'right brow']
        self.organs_landmarks = [list(range(0, 17)), list(range(48, 61)), list(range(27, 36)),
                            list(range(42, 48)), list(range(36, 42)), list(range(22, 27)), list(range(17, 22))]
        self.detector()
        self.add_face()

    def detector(self):
        faces = detector(self.img, 1)
        self.face = faces[0]
        # show the face
        # face = self.face
        # top, bottom, left, right = face.top(), face.bottom(), face.left(), face.right()
        # pt1, pt2 = (left, top), (right, bottom)
        # cv2.rectangle(self.img, pt1, pt2, (0, 0, 255))
        self.Points = np.array([[p.x, p.y] for p in predictor(self.img, self.face).parts()])
        self.Organs = {name: Organ(self.img, self.Points[landmarks], name)
                       for name, landmarks in zip(self.organs_name, self.organs_landmarks)}

    def add_face(self):
        face = self.face
        top, bottom, left, right = face.top(), face.bottom(), face.left(), face.right()
        # pt1, pt2 = (left, top), (right, bottom)
        # cv2.rectangle(self.img, pt1, pt2, (0, 0, 255))
        Points = np.array([[left, top], [right, bottom], [left, bottom], [right, top]])
        self.Organs['face'] = Face(self.img, Points, 'face')
