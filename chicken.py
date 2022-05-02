from re import M
import pixellib
from pixellib.instance import instance_segmentation

import cv2
import numpy as np
import imutils

model = instance_segmentation()
model.load_model("mask_rcnn_coco.h5") 

class Chicken:
    def __init__(self, filename, verbose = 0):
        self.filename = filename
        self.verbose = verbose

        self.chicken_area = self.get_chicken()
        self.magic_area = self.get_magic()

        self.ratio = 0 if (self.chicken_area == 0 or self.magic_area == 0) else self.chicken_area / self.magic_area

    def get_chicken(self):
        segmask, _ = model.segmentImage(self.filename, show_bboxes = False, extract_segmented_objects = True, output_image_name = self.filename + "seg.jpg" if self.verbose > 0 else None)
        
        class_ids = list(segmask["class_ids"])
        if 15 in class_ids: # bird class
            use_mask = class_ids.index(15)
        elif 17 in class_ids: # dog class for some reason
            use_mask = class_ids.index(17)
        else:
            return 0

        # get max area mask
        return np.count_nonzero(segmask["extracted_objects"][use_mask])
        # mask = [np.count_nonzero(mask) for mask in segmask["extracted_objects"]]
        # return mask[mask.index(max(mask))] if len(mask) > 0 else 1

    def get_magic(self):
        img = cv2.cvtColor(cv2.imread(self.filename), cv2.COLOR_BGR2HSV)
        img = imutils.resize(img, width=300)

        lower_range = np.array([110,50,50])
        upper_range = np.array([130,255,255])

        mask = cv2.inRange(img, lower_range, upper_range)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contour = [c for c in contours if cv2.contourArea(c) > 100]

        # download image with contour as filename+cont.jpg
        if self.verbose > 0:
            cv2.drawContours(img, [contour], -1, (0,255,0), 3)
            cv2.imwrite(self.filename + "segcont.jpg", img)

        if contour and len(contour) > 0:
            contour = max(contour, key=cv2.contourArea)
            return cv2.contourArea(contour)

        return 0