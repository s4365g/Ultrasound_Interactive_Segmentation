import os
import cv2
import numpy as np

from lesion_contours_detect import lesion_contours_andy

# image_path
img_path = os.path.join('ultrasound', 'Ultrasound_Info', 'hemangiomas', '07.jpg')

# read image
img_raw = cv2.imread(img_path)

# select ROIs function
ROIs = cv2.selectROIs("Select Rois", img_raw)

# print rectangle points of selected roi
print(ROIs)

# Crop selected roi ffrom raw image

# counter to save image with different name
crop_number = 0

# loop over every bounding box save in array "ROIs"
for rect in ROIs:
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]

    # crop roi from original image
    img_crop = img_raw[y1:y1 + y2, x1:x1 + x2]

    countour = lesion_contours_andy(img_crop, (x1, y1))
    cv2.drawContours(img_raw, countour, -1, (255, 0, 0), 1)
    # # show cropped image
    # cv2.imshow("crop" + str(crop_number), img_crop)
    #
    # # save cropped image
    # cv2.imwrite("crop" + str(crop_number) + ".jpg", img_crop)
    #
    # crop_number += 1

cv2.imshow("drawContours", img_raw)
# hold window
cv2.waitKey(0)