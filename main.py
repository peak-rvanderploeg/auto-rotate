
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2
import imutils

Image.MAX_IMAGE_PIXELS = None
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for filename in os.listdir('surrogates'):
        print(filename)
        img = cv2.imread("surrogates/" + filename)
        (B, G, R) = cv2.split(img)
        # Edit for fine tuning
        thresholding_lower_bound = 150 #THIS SHOULD PROBABLY BE 65

        _, threshold = cv2.threshold(B, thresholding_lower_bound, 255, cv2.THRESH_BINARY)
        pilthresh = Image.fromarray(threshold)

        # Edit these for fine tuning
        kernel_x=10
        kernel_y=10

        kernel = np.ones((kernel_x, kernel_y), np.uint8)
        threshold = cv2.erode(threshold, kernel)
        eroded = Image.fromarray(threshold)

        coords = np.column_stack(np.where(threshold > 0))
        angle = cv2.minAreaRect(coords)[-1]
        img_rot = imutils.rotate(img, -1 * (angle))
        pilim = Image.fromarray(cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB))
        pilim.show()
        pilim.save("rotated/" + "rotated-" +filename)
        print(angle)
        if (angle == 90):
            print("Warning, angle may not have been properly calculated")
