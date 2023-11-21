import numpy as np
import cv2 as cv
import os
import glob
import matplotlib.pyplot as plt
from numpy.random import uniform
import pdb

def find_color_values_using_trackbar(frame):

    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
 
    def nothing(x):
        pass

    cv.namedWindow("Trackbar") 
    cv.createTrackbar("LH", "Trackbar", 85, 255, nothing)
    cv.createTrackbar("LS", "Trackbar", 155, 255, nothing)
    cv.createTrackbar("LV", "Trackbar", 145, 255, nothing)
    cv.createTrackbar("UH", "Trackbar", 91, 255, nothing)
    cv.createTrackbar("US", "Trackbar", 255, 255, nothing)
    cv.createTrackbar("UV", "Trackbar", 255, 255, nothing)
    
    
    while True:

        l_h = cv.getTrackbarPos("LH", "Trackbar")
        l_s = cv.getTrackbarPos("LS", "Trackbar")
        l_v = cv.getTrackbarPos("LV", "Trackbar")
        u_h = cv.getTrackbarPos("UH", "Trackbar")
        u_s = cv.getTrackbarPos("US", "Trackbar")
        u_v = cv.getTrackbarPos("UV", "Trackbar")


        l = np.array([l_h, l_s, l_v])
        u = np.array([u_h, u_s, u_v])
        mask_table_hsv = cv.inRange(frame_hsv, l, u)        

        res = cv.bitwise_and(frame, frame, mask=mask_table_hsv)    
        cv.namedWindow("Frame", cv.WINDOW_NORMAL)
        cv.namedWindow("Mask", cv.WINDOW_NORMAL)
        cv.namedWindow("Res", cv.WINDOW_NORMAL)
        cv.imshow("Frame", frame)
        cv.imshow("Mask", mask_table_hsv)
        cv.imshow("Res", res)

        if cv.waitKey(25) & 0xFF == ord('q'):
                break
    cv.destroyAllWindows()

def main():
    path = '../date/imagini_auxiliare/'
    image_name = '02.jpg'
    img = cv.imread(path + image_name)
    find_color_values_using_trackbar(img)
    low_yellow = (15, 105, 105)
    high_yellow = (90, 255, 255)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
    cv.namedWindow("img_initial", cv.WINDOW_NORMAL)
    cv.namedWindow("mask_yellow_hsv", cv.WINDOW_NORMAL)
    cv.imshow('img_initial', img)
    cv.imshow('mask_yellow_hsv', mask_yellow_hsv)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
