import numpy as np
import cv2

# arg: gray1,gray2,两幅灰度图作为输入
# output：kp1,kp2是cv::keypoint的list，good是cv::DMatch的list
def sift(gray1, gray2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)  # DMatch

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    return kp1, kp2, good
