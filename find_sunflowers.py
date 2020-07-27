import torch
import torch.nn
import cv2 as cv
import numpy as np
from matplotlib.pyplot import imread

from SunflowerFullConnectedClassificator import SunflowerFullConnectedClassificator

image = cv.imread('sunflowers/test1.png')
SIZE = (15, 15)
print(np.mean(image))
img = np.where(image > 160, 255, 0).astype("uint8")
imgray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
img_res = image.copy()
contours = list(filter(lambda x: cv.arcLength(x, True) > 5, contours))

net = SunflowerFullConnectedClassificator(675, 1)
net.load_state_dict(torch.load("sunflowerClassifier.pth"))
net.eval()
m= torch.nn.Sigmoid()
count = 1
for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    if w*h > 500:
        continue
    cv.rectangle(img_res, (x, y), (x + w, y + h), (0, 255, 0), 2)
    crop_img = image[y:y + h, x:x + w]
    crop_img = cv.resize(crop_img, SIZE)
    crop_img = (cv.cvtColor(crop_img, cv.COLOR_BGR2RGB)/255)
    crop_tensor=torch.from_numpy(np.reshape(crop_img, (1, crop_img.shape[0] * crop_img.shape[1] * 3))).float().to("cpu")
    res = (m(net(crop_tensor)) >= 0.3).item()
    if res == 1:
        cv.rectangle(img_res, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv.rectangle(img_res, (x, y), (x + w, y + h), (0, 0, 255), 2)
    count += 1
cv.drawContours(img_res, contours, -1, (0,255,0), 1)

cv.imshow("res", img_res)
cv.imwrite("sunFlowersTest/original.png", image)
cv.imwrite("sunFlowersTest/res.png", img_res)
cv.imwrite("sunFlowersTest/contrast.png", img)
cv.waitKey(0)

# closing all open windows
cv.destroyAllWindows()