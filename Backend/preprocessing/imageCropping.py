import cv2
import os

class cropImage:

    def getLargestBBoxArea(img, val_thresh):

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray_img, val_thresh, 255, 0)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        largest_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_area:
                largest_area = area
                x, y, w, h = cv2.boundingRect(contour)

        return (x, y, w, h)

    def crop_image(img, bbox):

        (x, y, w, h) = bbox
        return img[y:y+h, x:x+w]

    def draw_bounding_box(img, bbox, thickness):

        (x, y, w, h) = bbox
        reshaped_image = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 255, 0), thickness)

        return reshaped_image