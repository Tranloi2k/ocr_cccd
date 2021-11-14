# import the necessary packages
from imutils.contours import sort_contours
import numpy as np
import pytesseract
import imutils
import sys
import cv2

# construct the argument parser and parse the arguments

def ocr_cccd(image):
    # load the input image, convert it to grayscale, and grab its dimensions
    image = imutils.resize(image, width=1000)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # initialize a rectangular and square structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    # smooth the image using a 3x3 Gaussian blur and then apply a
    # blackhat morpholigical operator to find dark regions on a light background
    #gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    thres = cv2.threshold(blackhat.copy(), 40, 250, cv2.THRESH_BINARY)[1]
    #edged = cv2.Canny(gray, 50, 200)




    #rect = cv2.getStructuringElement(cv2.MORPH_RECT, (2,1))
    #thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, rect)

    #-c tessedit_char_whitelist=0123456789
    options = "-l eng --psm 11 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(thres, config=options)

    return text



