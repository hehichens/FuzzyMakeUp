'''
main function
'''

import dlib
import cv2
from MakeUp import *

import numpy as np
from Constant import *

if __name__ == "__main__":
    img = cv2.imread(ImagePath)
    MU = MakeUp(img)
    MU.Organs['jaw'].Whitening(0.25) # 这里有一个bug, 调用化妆方法之后再调用另一个化妆方法，　img 会重新刷新
    MU.Organs['jaw'].Smooth()
    MU.Organs['mouth'].Brightening()
    MU.Organs['mouth'].Smooth()
    MU.Organs['mouth'].Whitening(0.75)  # 在同一个Organ对象下调用化妆方法不会刷新
    cv2.imshow("img3", img)
    cv2.waitKey(0)