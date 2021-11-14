# import the necessary packages
import numpy as np
import imutils
import glob
import cv2

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-t", "--template", required=True, help="Path to template image")
#ap.add_argument("-i", "--images", required=True,
#                help="Path to images where template will be matched")

#args = vars(ap.parse_args())
# load the image image, convert it to grayscale, and detect edges\
def matching(input):
    template = cv2.imread("cccd2.jpg")
    template = imutils.resize(template, height=30)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 40, 200)
    (tH, tW) = template.shape[:2]

    # loop over the images to find the template in
        # load the image, convert it to grayscale, and initialize the
        # bookkeeping variable to keep track of the matched region
    image = input.copy()
    image = imutils.resize(image, width=1200)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None
    a = False
    # loop over the scales of the image
    for scale in np.linspace(0.1, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)

        locations = np.where(result >= 0.2)
        locations = list(zip(*locations[::-1]))

        if locations:
            a=True
            for loc in locations:
                top_left = loc
                bottom_right = (top_left[0]+tW, top_left[1]+tH)
                cv2.rectangle(resized, top_left, bottom_right, (255, 0, 0), 3)

    return a
