import cv2
import numpy as np

def FT(gray_img):
    '''
    input the gray image:
    '''
    Histgram = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    Temp = [i for i in range(len(Histgram)) if Histgram[i] != 0]
    first, last = Temp[0], Temp[-1]
    if(last-first <= 1):
        return first

    S = np.zeros(last+1)
    W = np.zeros(last+1)
    S[0] = Histgram[0]
    for i in range(first, last+1):
        S[i] = S[i-1] + Histgram[i]
        W[i] = W[i-1] + i * Histgram[i]

    Smu = np.zeros(last-first+1)
    for i in range(1, len(Smu)):
        mu = 1/(1+i/(last-first))
        Smu[i] = -mu*np.log(mu) - (1 - mu)*np.log(1-mu)
    BestEntropy = np.inf
    for i in range(first, last):
        Entropy = 0

        mu = int(W[i] // S[i])
        for X in range(first, i+1):
            Entropy += Smu[np.abs(X - mu)]*Histgram[X]

        mu = int((W[last] - W[i])//(S[last] - S[i]))
        for X in range(i+1, last+1):
            Entropy += Smu[np.abs(X - mu)]*Histgram[X]

        if BestEntropy > Entropy:
            BestEntropy = Entropy
            Threshold = i

    ret, thresh = cv2.threshold(gray_img, Threshold, 255, cv2.THRESH_BINARY)
    return thresh

if __name__ == '__main__':
    '''
    test Fuzzy Threshold
    '''
    img = cv2.imread("/home/hichens/Datasets/pic/11.jpg", 0)  # "/home/hichens/Datasets/xieshi_test/1.jpg"

    thresh = FT(img)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    
    cv2.imshow("thresh", thresh)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()