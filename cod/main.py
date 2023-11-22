import cv2 as cv
import numpy as np
import os

def showImage(image):
    cv.namedWindow("Imagine", cv.WINDOW_NORMAL)
    cv.imshow("Imagine", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def filterImage(img):
    l_h = 85
    l_s = 155
    l_v = 145
    u_h = 100
    u_s = 255
    u_v = 255
    frame_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    l = np.array([l_h, l_s, l_v])
    u = np.array([u_h, u_s, u_v])
    mask_table_hsv = cv.inRange(frame_hsv, l, u)        
    res = cv.bitwise_and(img, img, mask=mask_table_hsv)    
    res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    res = cv.medianBlur(res,5)
    res = cv.GaussianBlur(res, (0, 0), 7) 
    _, res = cv.threshold(res, 10, 255, cv.THRESH_BINARY)

    # TODO: try Canny Edge
    return res
    
def getBoardCorners():
    path = '../date/imagini_auxiliare/'
    image_name = '01.jpg'
    points = []

    image = cv.imread(path + image_name)

    contours, _ = cv.findContours(filterImage(image), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        epsilon = 0.02 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            board_corners = approx.reshape((4, 2))
            board_corners[0][0] += 30
            board_corners[3][0] += 30
            board_corners[3][1] += 30
            board_corners[2][1] += 30
            return board_corners

def getBoard(image, board_corners):
    height, width = image.shape[:2]
    board_position = np.array([board_corners[1], board_corners[0], board_corners[3], board_corners[2]], dtype="float32")
    destination_board_position = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    M = cv.getPerspectiveTransform(board_position, destination_board_position)
    result = cv.warpPerspective(image, M, (width, height))

    return result

def main():
    image_name = '5_20.jpg'
    path = '../date/antrenare/'
    image = cv.imread(path + image_name)

    board_corners = getBoardCorners()
    image = getBoard(image, board_corners)

    showImage(image)

if __name__ == "__main__":
    main()
