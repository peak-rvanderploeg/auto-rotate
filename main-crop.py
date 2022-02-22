
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2
import imutils

Image.MAX_IMAGE_PIXELS = None
import os


# This can be any number between 0-255 but should probably be ~65
thresholding_lower_bound = 65



def find_min_rect(img):
    (B, G, R) = cv2.split(img)

    _, threshold = cv2.threshold(B, thresholding_lower_bound, 255, cv2.THRESH_BINARY)
    pilthresh = Image.fromarray(threshold)

    # Edit these for fine tuning
    kernel_x = 10
    kernel_y = 10

    kernel = np.ones((kernel_x, kernel_y), np.uint8)
    threshold = cv2.erode(threshold, kernel)
    eroded = Image.fromarray(threshold)

    coords = np.column_stack(np.where(threshold > 0))
    rect = cv2.minAreaRect(coords)
    return rect

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for filename in os.listdir('surrogates'): #This script will run over every file in surrogates
        print(filename)
        img = cv2.imread("surrogates/" + filename)

        # Creates a thresholding mask on the input img and then finds the minimum rectangle surrounded the
        # thresholded area
        rot_rect = find_min_rect(img)

        # Uses the angle of the output rectangle as a measurement of how rotated the surrogate is, then rotates it back.
        img_rot = imutils.rotate(img, -1 * rot_rect[2])
        plt.imshow(Image.fromarray(cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB)))
        plt.title("Rotated")
        plt.show()

        #Finds the rectangle surrounding the newly rotated surrogate
        crop_rect = find_min_rect(img_rot)

        #Captures the x,y coordinates of the corners of said rectangle
        [A,B,C,D] = cv2.boxPoints(crop_rect)

        #Crops the image at using the corners of the rectangle
        cropped = img_rot[(int(A[0])-10):(int(B[0])+10),(int(A[1])-10):(int(C[1])+10)]

        #Saves the cropped image
        cropped = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        plt.imshow(cropped)
        plt.title("Cropped")
        plt.show()
        cropped.save("rotated/"+"rotated-"+filename)
