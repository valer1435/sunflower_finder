import numpy as np
import cv2 as cv

image = cv.imread('sunflowers/field_01_contrast.jpg')
image_original = cv.imread('sunflowers/field_01.jpg')

SIZE = (15, 15)
PADDING = 3

imgray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
img_res = image.copy()

contours = list(filter(lambda x: cv.arcLength(x, True) > 20, contours))
count=1
for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    cv.rectangle(img_res, (x, y), (x + w+PADDING, y + h+PADDING), (0, 255, 0), 2)
    crop_img = image_original[y:y + h, x:x + w]
    crop_img = cv.resize(crop_img, SIZE)

    cv.imwrite(f"sunflowerFieldsContours/sunflower_{count}.png", crop_img)
    count += 1
cv.drawContours(img_res, contours, -1, (0,255,0), 1)

cv.imshow("res", img_res)

cv.waitKey(0)

# closing all open windows
cv.destroyAllWindows()