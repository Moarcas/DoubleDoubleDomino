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

board_track = [-1, 1, 2, 3, 4, 5, 6, 0, 2, 5, 3, 4, 6, 2, 2, 0,
          3, 5, 4, 1, 6, 2, 4, 5, 5, 0, 6, 3, 4, 2, 0, 1, 
          5, 1, 3, 4, 4, 4, 5, 0, 6, 3, 5, 4, 1, 3, 2, 0,
          0, 1, 1, 2, 3, 6, 3, 5, 2, 1, 0, 6, 6, 5, 2, 1,
          2, 5, 0, 3, 3, 5, 0, 6, 1, 4, 0, 6, 3, 5, 1, 4,
          2, 6, 2, 3, 1, 6, 5, 6, 2, 0, 4, 0, 1, 6, 4, 4,
          1, 6, 6, 3, 0]

path_read = '../date/evaluare/fake_test/'
path_write = '../date/evaluare/fisiere_solutie/351_Moarcas_Cosmin/'
#path_read = '../date/antrenare/'
#path_write = '../date/351_Moarcas_Cosmin/'

number_games = 1
number_moves = 20

def showImage(image):
    cv.namedWindow("Imagine", cv.WINDOW_NORMAL)
    cv.imshow("Imagine", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Filter to get the game board
def filterImage(image):
    frame_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    l = np.array([20, 20, 0])
    u = np.array([120, 255, 255])
    mask_table_hsv = cv.inRange(frame_hsv, l, u)        
    image = cv.bitwise_and(image, image, mask=mask_table_hsv)    
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY)
    kernel = np.ones((3, 3),np.uint8)
    image = cv.erode(image, kernel, iterations=4)
    image = cv.dilate(image, kernel, iterations=7)
    return image

# Filter to get domios from the board 
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
    showImage(image)
    return image
    
def getBoardCorners(image):
    image_filtered = filterImage(image)
    contours, _ = cv.findContours(image_filtered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area1= 0
    top_left1 = 0
    top_right1 = 0
    bottom_left1 = 0
    bottom_right1 = 0
    max_area2 = 0
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
            contour_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
            if  contour_area > max_area1:
                max_area2 = max_area1                
                top_left2 = top_left1
                bottom_right2 = bottom_right1
                top_right2 = top_right1
                bottom_left2 = bottom_left1
                max_area1 = contour_area
                top_left1 = possible_top_left
                bottom_right1 = possible_bottom_right
                top_right1 = possible_top_right
                bottom_left1 = possible_bottom_left
            elif contour_area > max_area2:
                max_area2 = contour_area
                top_left2 = possible_top_left
                bottom_right2 = possible_bottom_right
                top_right2 = possible_top_right
                bottom_left2 = possible_bottom_left

    return top_left2, top_right2, bottom_right2, bottom_left2


def getBoard(image):
    height, width = image.shape[:2]
    top_left, top_right, bottom_right, bottom_left = getBoardCorners(image)
    board_position = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    destination_board_position = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    M = cv.getPerspectiveTransform(board_position, destination_board_position)
    result = cv.warpPerspective(image, M, (width, height))

    return result

def getPositionByLine(line):
    return 269 * line + 30

def getPositionByColumn(column):
    return 202* column + 22

def getPiecePosition(image, last_image):
    difference_image = filterTable(image) - filterTable(last_image)
    showImage(difference_image)
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
    half_domino = cv.GaussianBlur(half_domino, (9, 9), 2)
    half_domino = np.clip(1.5 * half_domino + 150, 0, 255).astype(np.uint8)

    circles = cv.HoughCircles(half_domino, cv.HOUGH_GRADIENT, 1, 40,
                               param1=100, param2=16,
                               minRadius=25, maxRadius=32)
    nr_dots = 0
    if circles is not None:
        nr_dots = circles.shape[1]
    return nr_dots


def processGames():
    empty_board = getBoard(cv.imread('../date/imagini_auxiliare/01.jpg'))

    for joc in range(1, number_games + 1):
        file_players_order = open(path_read + f'{joc}_mutari.txt')
        last_image = empty_board
        points = [0, 0]
        for i in range(1, number_moves + 1):
            current_player = int(file_players_order.readline().split()[1][-1]) - 1

            if i < 10:
                image_name = f"{joc}_0{i}.jpg"
            else:
                image_name = f"{joc}_{i}.jpg"
            
            image = cv.imread(path_read + image_name)
            image = getBoard(image)

            square1_position, square2_position = getPiecePosition(image, last_image)

            square1_nr_dots = getNumberOfDots(image, *square1_position)
            square2_nr_dots = getNumberOfDots(image, *square2_position)
            
            move_points = points_matrix[square1_position[0]][square1_position[1]] + points_matrix[square2_position[0]][square2_position[1]] 
            move_points += int(square1_nr_dots == square2_nr_dots) * move_points 

            # check for bonus points
            if square1_nr_dots == board_track[points[current_player]] or square2_nr_dots == board_track[points[current_player]]:
                move_points += 3
            if square1_nr_dots == board_track[points[1 - current_player]] or square2_nr_dots == board_track[points[1 - current_player]]:
                points[1 - current_player] += 3
            
            points[current_player] += move_points

            line_square1 = square1_position[0] + 1
            line_square2 = square2_position[0] + 1
            column_square1 = chr(ord('A') + square1_position[1])
            column_square2 = chr(ord('A') + square2_position[1])
            
            file_name = image_name[:-3] + 'txt'

            with open(path_write + file_name, 'w') as file:
                file.write(str(line_square1) + str(column_square1) + ' ' + str(square1_nr_dots) + '\n')
                file.write(str(line_square2) + str(column_square2) + ' ' + str(square2_nr_dots) + '\n')
                file.write(str(move_points))

            last_image = image

def main():
    processGames()

if __name__ == "__main__":
    main()
