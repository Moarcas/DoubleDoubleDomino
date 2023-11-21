import cv2 as cv
import numpy as np
import os

def showImage(image):
    cv.namedWindow("Imagine", cv.WINDOW_NORMAL)
    cv.imshow("Imagine", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def filterImage(image):
    frame_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    l = np.array([85, 90, 110])
    u = np.array([100, 255, 255])
    mask_table_hsv = cv.inRange(frame_hsv, l, u)        
    res = cv.bitwise_and(image, image, mask=mask_table_hsv)    
    res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    _, res = cv.threshold(res, 10, 255, cv.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    res = cv.morphologyEx(res, cv.MORPH_OPEN, kernel)
    res = cv.GaussianBlur(res, (5, 5), 0)
    showImage(res)
    return res
    
def getBoardCorners():
    path = '../date/imagini_auxiliare/'
    image_name = '01.jpg'

    image = cv.imread(path + image_name)
    filtered_image = filterImage(image)
    contours, _ = cv.findContours(filtered_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, -1, (0, 255, 0), 2)
    showImage(image)
    for cnt in contours:
        epsilon = 0.02 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            board_corners = approx.reshape((4, 2))
            return board_corners

def getBoard(image, board_corners):
    height, width = image.shape[:2]
    board_position = np.array([board_corners[1], board_corners[0], board_corners[3], board_corners[2]], dtype="float32")
    destination_board_position = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    M = cv.getPerspectiveTransform(board_position, destination_board_position)
    result = cv.warpPerspective(image, M, (width, height))

    return result

def filterBoard(image):
    l_h = 87
    l_s = 0
    l_v = 234
    u_h = 255
    u_s = 255
    u_v = 255
    frame_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    l = np.array([l_h, l_s, l_v])
    u = np.array([u_h, u_s, u_v])
    mask_table_hsv = cv.inRange(frame_hsv, l, u)        
    image = cv.bitwise_and(image, image, mask=mask_table_hsv)    
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, res = cv.threshold(image, 10, 255, cv.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)
    image = cv.morphologyEx(res, cv.MORPH_OPEN, kernel)

    return image

def readFile(filepath):
    player_order = []
    with open(filepath, 'r') as file:
        for line in file:
            player_order.append(int(line[-2]))
    return player_order

def readGame(number_game):
    folder_path = '../date/antrenare/'
    images = []
    player_order = readFile(folder_path + number_game + '_mutari.txt')

    # TODO: delete this
    path = '../date/imagini_auxiliare/01.jpg'
    return np.array([cv.imread(path)]), player_order

     
    for filename in os.listdir(folder_path):
        if filename[0] == number_game:
            if filename.endswith('.jpg'):
                file_path = os.path.join(folder_path, filename)
                image = cv.imread(file_path)
                images.append(image)
    return np.array(images), player_order


def main():
    number_games = 5
    images, player_order = readGame('5')
    image = getBoard(images[0], getBoardCorners())
    #showImage(image)

if __name__ == "__main__":
    main()
