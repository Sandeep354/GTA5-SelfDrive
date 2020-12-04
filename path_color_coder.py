
# https://imagecolorpicker.com/en/
# Check color of the path (or anything else)

'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("minimap_example.png")
#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #hsv image

lower_purple = (255,255,255)
upper_purple = (255,0,205)

mask = cv2.inRange(img, lower_purple, upper_purple)
lane = cv2.bitwise_and(img, img, mask)

cv2.imwrite("laneOnly.png", lane)

'''

'''
        THIS CODE WORKS BETTER (HSV > RGB)
img = cv2.imread("minimap_example.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#lower_purple = (100, 45, 120) # you might want to refine these
#upper_purple = (260, 105, 200) # you might want to refine these

lower_purple = (136,167,155)
upper_purple = (136,167,243)

mask = cv2.inRange(hsv, lower_purple, upper_purple)

lane = cv2.bitwise_and(img, img, mask=mask)
#gray = cv2.cvtColor(lane, cv2.COLOR_RGB2GRAY)
cv2.imwrite("lane.png", lane)

'''
import cv2
import numpy as np

# Only highlight path --> Process the image
def lane_finder(image):

    #image = cv2.imread(image) --> Not needed during recording as it is already satisfied
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_purple = (136,167,155)
    upper_purple = (136,167,243)

    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    lane = cv2.bitwise_and(image, image, mask=mask)

    #Canny and Blurring 
    #lane = cv2.cvtColor(lane, cv2.COLOR_HSV2RGB)	
    #lane = cv2.cvtColor(lane, cv2.COLOR_RGB2GRAY)
    lane = cv2.Canny(lane, threshold1=200, threshold2=300)
    lane = cv2.GaussianBlur(lane, (3,3), 0 )

    # Doing edge detection of path will compute less than doing whole path detection

    return lane



