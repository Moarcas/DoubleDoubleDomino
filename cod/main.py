import cv2 as cv
import numpy as np
import os

points_matrix = [
    [5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5],
    [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
    [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
    [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
    [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
    [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0 ,0],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0 ,0],
    [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
    [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
    [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
    [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
    [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
    [5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5],
]

traseu = [-1, 1, 2, 3, 4, 5, 6, 0, 2, 5, 3, 4, 6, 2, 2, 0
          3, 5, 4, 1, 6, 2, 4, 5, 5, 0, 6, 3, 4, 2, 0, 1, 
          5, 1, 3, 4, 4, 4, 5, 0, 6, 3, 5, 4, 1, 3, 2, 0,
          0, 1, 1, 2, 3, 6, 3, 5, 2, 1, 0, 6, 6, 5, 2, 1
          2, 5, 0, 3, 3, 5, 0, 6, 1, 4, 0, 6, 3, 5, 1, 4
          2, 6, 2, 3, 1, 6, 5, 6, 2, 0, 4, 0, 1, 6, 4, 4,
          1, 6, 6, 3, 0]

def showImage(image):
    cv.namedWindow("Imagine", cv.WINDOW_NORMAL)
    cv.imshow("Imagine", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def filterImage(image):
    frame_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    l = np.array([20, 20, 0])
    u = np.array([120, 255, 255])
    mask_table_hsv = cv.inRange(frame_hsv, l, u)        
    image = cv.bitwise_and(image, image, mask=mask_table_hsv)    
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    image = cv.erode(image, kernel, iterations=3)
    image = cv.dilate(image, kernel, iterations=5)
    return image
    
def getBoardCorners(image):
    image_filtered = filterImage(image)
    contours, _ = cv.findContours(image_filtered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    top_left2 = 0
    top_right2 = 0
    bottom_left2 = 0
    bottom_right2 = 0
   
    for i in range(len(contours)):
        if(len(contours[i]) > 3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis = 1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                top_left = top_left2
                bottom_right = bottom_right2
                top_right = top_right2
                bottom_left = bottom_left2
                top_left2 = possible_top_left
                bottom_right2 = possible_bottom_right
                top_right2 = possible_top_right
                bottom_left2 = possible_bottom_left
    return top_left, top_right, bottom_right, bottom_left


def getBoard(image):
    height, width = image.shape[:2]
    top_left, top_right, bottom_right, bottom_left = getBoardCorners(image)
    board_position = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    destination_board_position = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    M = cv.getPerspectiveTransform(board_position, destination_board_position)
    result = cv.warpPerspective(image, M, (width, height))

    return result

def drawLines(image):
    for line in range(28, image.shape[0], 268):
        cv.line(image, (0 ,line), (image.shape[1] ,line), (0, 0, 255), 3)

    for column in range(22, image.shape[1], 202):
        cv.line(image, (column ,0), (column, image.shape[0]), (0, 0, 255), 3)


def saveBoardImages():
    path_read = '../date/antrenare/'
    path_write = '../date/antrenare_tabla_joc/'
    
    for joc in range(1, 6):
        for i in range(1, 21):
            if i < 10:
                image_name = f"{joc}_0{i}.jpg"
            else:
                image_name = f"{joc}_{i}.jpg"
                
            image = cv.imread(path_read + image_name)
            image = getBoard(image)
            cv.imwrite(path_write + image_name, image)

def filterTable(image):
    frame_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    l = np.array([90, 20, 220])
    u = np.array([150, 115, 255])
    mask_table_hsv = cv.inRange(frame_hsv, l, u)        
    image = cv.bitwise_and(image, image, mask=mask_table_hsv)    
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, image = cv.threshold(image, 150, 255, cv.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    image = cv.medianBlur(image, 25)
    return image

def getPositionByLine(line):
    return 268 * line + 28

def getPositionByColumn(column):
    return 202* column + 22

def getPiecePosition(image, last_image):
    difference_image = filterTable(image) - filterTable(last_image)
    max_mean1 = 30
    max_mean2 = 30
    point1 = (0, 0)
    point2 = (0, 0)

    for line in range(15):
        for column in range(15):
            line_position_start = getPositionByLine(line)
            line_position_end = line_position_start + 268
            column_position_start = getPositionByColumn(column)
            column_position_end = column_position_start + 202

            top_left = (column_position_start, line_position_start)
            top_right = (column_position_start, line_position_end)
            bottom_left = (column_position_end, line_position_start)
            bottom_right = (column_position_end, line_position_end)

            mean = np.mean(difference_image[line_position_start:line_position_end, column_position_start:column_position_end])

            if mean > max_mean1:
                max_mean2 = max_mean1
                point2 = point1
                max_mean1 = mean
                point1 = (line, column)
            elif mean > max_mean2:
                max_mean2 = mean
                point2 = (line, column)

    point1, point2 = sorted([point1, point2])
    return (point1, point2)

def getNumberOfDots(image, line, column):
    line_start = getPositionByLine(line)
    line_end = line_start + 268
    column_start = getPositionByColumn(column)
    column_end = column_start+ 202
    half_domino = image[line_start:line_end, column_start:column_end]
    x = half_domino
    half_domino = cv.cvtColor(half_domino, cv.COLOR_BGR2GRAY)
    half_domino = cv.medianBlur(half_domino, 15)
    half_domino = np.clip(1.5 * half_domino + 150, 0, 255).astype(np.uint8)

    circles = cv.HoughCircles(half_domino, cv.HOUGH_GRADIENT, 1, 50,
                               param1=100, param2=15,
                               minRadius=20, maxRadius=30)
    #if circles is not None:
    #    circles = np.uint16(np.around(circles))
    #    for i in circles[0, :]:
    #        center = (i[0], i[1])
    #        # circle center
    #        cv.circle(x, center, 1, (0, 100, 100), 3)
    #        # circle outline
    #        radius = i[2]
    #        cv.circle(x, center, radius, (255, 0, 255), 3)
    #showImage(x)
    nr_dots = 0
    if circles is not None:
        nr_dots = circles.shape[1]
    return nr_dots


def processGames():
    path_read = '../date/antrenare_tabla_joc/'
    path_write = '../date/351_Moarcas_Cosmin/'
    last_image_ = getBoard(cv.imread('../date/imagini_auxiliare/01.jpg'))

    for joc in range(1, 6):
        last_image = last_image_
        for i in range(1, 21):
            if i < 10:
                image_name = f"{joc}_0{i}.jpg"
            else:
                image_name = f"{joc}_{i}.jpg"
            
            file_name = image_name[:-3] + 'txt'

            image = cv.imread(path_read + image_name)
            point1, point2 = getPiecePosition(image, last_image)

            nr_dots_point1 = getNumberOfDots(image, *point1)
            nr_dots_point2 = getNumberOfDots(image, *point2)
        
            line_point1 = point1[0] + 1
            line_point2 = point2[0] + 1
            column_point1 = chr(ord('A') + point1[1])
            column_point2 = chr(ord('A') + point2[1])

            with open(path_write + file_name, 'w') as file:
                file.write(str(line_point1) + str(column_point1) + ' ' + str(nr_dots_point1) + '\n')
                file.write(str(line_point2) + str(column_point2) + ' ' + str(nr_dots_point2))

            last_image = image


def processGamesDebug():
    path_read = '../date/antrenare_tabla_joc/'
    path_write = '../date/351_Moarcas_Cosmin/'
    last_image = cv.imread(path_read + '1_12.jpg')
    last_image = getBoard(cv.imread('../date/imagini_auxiliare/01.jpg'))
    image = cv.imread(path_read + '5_01.jpg')
    file_name = '5_01.txt'

    point1, point2 = getPiecePosition(image, last_image)

    nr_dots_point1 = getNumberOfDots(image, *point1)
    nr_dots_point2 = getNumberOfDots(image, *point2)

    line_point1 = point1[0] + 1
    line_point2 = point2[0] + 1
    column_point1 = chr(ord('A') + point1[1])
    column_point2 = chr(ord('A') + point2[1])

    with open(path_write + file_name, 'w') as file:
        file.write(str(line_point1) + str(column_point1) + ' ' + str(nr_dots_point1) + '\n')
        file.write(str(line_point2) + str(column_point2) + ' ' + str(nr_dots_point2))


def main():
    #processGames()

if __name__ == "__main__":
    main()
