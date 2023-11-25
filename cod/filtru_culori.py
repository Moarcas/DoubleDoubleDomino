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
    cv.createTrackbar("LH", "Trackbar", 90, 255, nothing)
    cv.createTrackbar("LS", "Trackbar", 0, 255, nothing)
    cv.createTrackbar("LV", "Trackbar", 235, 255, nothing)
    cv.createTrackbar("UH", "Trackbar", 150, 255, nothing)
    cv.createTrackbar("US", "Trackbar", 115, 255, nothing)
    cv.createTrackbar("UV", "Trackbar", 255, 255, nothing)
    cv.createTrackbar("Threshold", "Trackbar", 150, 255, nothing)
    
    while True:
        l_h = cv.getTrackbarPos("LH", "Trackbar")
        l_s = cv.getTrackbarPos("LS", "Trackbar")
        l_v = cv.getTrackbarPos("LV", "Trackbar")
        u_h = cv.getTrackbarPos("UH", "Trackbar")
        u_s = cv.getTrackbarPos("US", "Trackbar")
        u_v = cv.getTrackbarPos("UV", "Trackbar")
        threshold = cv.getTrackbarPos("Threshold", "Trackbar")

        l = np.array([l_h, l_s, l_v])
        u = np.array([u_h, u_s, u_v])
        mask_table_hsv = cv.inRange(frame_hsv, l, u)        

        image = cv.bitwise_and(frame, frame, mask=mask_table_hsv)    

        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, image = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)

        cv.namedWindow("Frame", cv.WINDOW_NORMAL)
        cv.namedWindow("Image", cv.WINDOW_NORMAL)
        cv.imshow("Frame", frame)
        cv.imshow("Image", image)

        if cv.waitKey(25) & 0xFF == ord('q'):
                break
    cv.destroyAllWindows()

def main():
    path = '../date/antrenare_tabla_joc/'
    image_name = "5_20.jpg"
    img = cv.imread(path + image_name)
    find_color_values_using_trackbar(img)
    return
    for i in range(1, 10):
        image_name = "5_0" + str(i) + ".jpg"
        img = cv.imread(path + image_name)
        find_color_values_using_trackbar(img)
    return
    for i in range(10, 21):
        image_name = "1_" + str(i) + ".jpg"
        img = cv.imread(path + image_name)
        find_color_values_using_trackbar(img)

    return

if __name__ == '__main__':
    main()
