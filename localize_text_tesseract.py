#python localize_text_tesseract.py  --front C:\Users\Admin\PycharmProjects\ORC\BeVietnam-Medium.ttf
# --image C:\Users\Admin\PycharmProjects\ORC\images\cccd2.jpg

# import the necessary packages
from pytesseract import Output
from helper.blur_and_threshold import blur_and_threshold
import pytesseract
import argparse
import cv2
import imutils
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image to be OCR'd")
ap.add_argument("-c", "--min-conf", type=int, default=50, help="mininum confidence value to filter weak text detection")
ap.add_argument("-p", "--psm", type=int, default=11, help="Tesseract PSM mode")
ap.add_argument("-f", "--front", required=True, help="path to front")
args = vars(ap.parse_args())



# load the input image, convert it from BGR to RGB channel ordering,
# and use Tesseract to localize each area of text in the input image
image = cv2.imread(args["image"])
image = imutils.resize(image, width=800)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# initialize a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 7))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))

# smooth the image using a 3x3 Gaussian blur and then apply a
# blackhat morpholigical operator to find dark regions on a light background
#gray = cv2.GaussianBlur(gray, (3, 3), 0)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
#blackhat = cv2.dilate(blackhat, (3, 3), 1)

thres = cv2.threshold(blackhat.copy(), 40, 250, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


options = "-l vie7 --psm {}".format( args["psm"])
#rect = cv2.getStructuringElement(cv2.MORPH_RECT, (2,1))
#thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, rect)

final = blur_and_threshold(gray)
blackhat = cv2.morphologyEx(final, cv2.MORPH_BLACKHAT, rectKernel)

results = pytesseract.image_to_data(final, config=options, output_type=Output.DICT)

# loop over each of the individual text localizations
for i in range(0, len(results["text"])):

    # extract the bounding box coordinates of the text region from the current result
    x = results["left"][i]
    y = results["top"][i]
    w = results["width"][i]
    h = results["height"][i]

    # extract the OCR text itself along with the confidence of the text localization
    text = results["text"][i]
    conf = int(results["conf"][i])

    # filter out weak confidence text localizations
    if conf > args["min_conf"]:

        # display the confidence and text to our terminal
        print("Confidence: {}".format(conf))
        print("Text: {}".format(text))
        print("")

        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw a bounding box around the text along with the text itself
        #text = cleanup_text(text)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

        fontpath = args["front"]
        font = ImageFont.truetype(fontpath, 18)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        draw.text((x+10, y-30), text, font=font, fill=(0, 0, 255))
        image = np.array(image)

# show the output image
cv2.imshow("", blackhat)
cv2.imshow("Image", imutils.resize(image, width=1000))
cv2.waitKey(0)


