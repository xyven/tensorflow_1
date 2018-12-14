#_*_ coding:UTF-8 _*_

import sys
import numpy as np
import cv2
import math
#import self-define packages
from RadarPreprocessor import Make36AnglePNG

#Make36AnglePNG(sys.argv[1])

def CalculateIOU(obboxes, pbboxes):
    image = np.zeros((460, 460, 1), dtype=np.uint8)
    im = np.zeros(image.shape[:2], dtype="uint8")
    im1 = np.zeros(image.shape[:2], dtype="uint8")
    original_grasp_mask = cv2.fillPoly(im, [obboxes], 255)
    prediction_grasp_mask = cv2.fillPoly(im1, [pbboxes], 255)
    masked_and = cv2.bitwise_and(original_grasp_mask, prediction_grasp_mask, mask=im)
    masked_or = cv2.bitwise_or(original_grasp_mask, prediction_grasp_mask)

    or_area = np.sum(np.float32(np.greater(masked_or, 0))) #或操作，得到并集
    and_area = np.sum(np.float32(np.greater(masked_and, 0))) #与操作，得到交集
    IOU = and_area / or_area
    return or_area, and_area, IOU

def CaluculateBoundingRect(obboxes):
    image = np.zeros((460,460,3), dtype=np.uint8)
    im = np.zeros(image.shape[:2], dtype="uint8")
    original_grasp_mask = cv2.fillPoly(image, [obboxes], 255)
    x, y, w, h= cv2.boundingRect(obboxes)

    rect = cv2.minAreaRect(points)
    print(rect)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = np.append(box, box[0])
    box = box.reshape(-1,2)
    print(box)

    cv2.drawContours(image, [box], 0, (255, 255, 0), 2)
    print(list(map(tuple, box)))

    print(x,y,w,h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('',image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def GetGoncateBounding(obboxes):
    image = np.zeros((460, 460, 3), dtype=np.uint8)
    hull = cv2.convexHull(obboxes, returnPoints=True)
    cv2.drawContours(image, [hull], 0, (0, 0, 255), 2)
    cv2.imshow('', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def rotatecordiate(angle, rect):
    angle=angle*math.pi/180
    n=1
    m=1
    def onepoint(x,y):
        # X = x*math.cos(angle) - y*math.sin(angle)-0.5*n*math.cos(angle)+0.5*m*math.sin(angle)+0.5*n
        # Y = y*math.cos(angle) + x*math.sin(angle)-0.5*n*math.sin(angle)-0.5*m*math.cos(angle)+0.5*m
        X = x * math.cos(angle) - y * math.sin(angle) - 0.5 * n * math.cos(angle) + 0.5 * m * math.sin(angle) + 0.5 * n
        Y = y * math.cos(angle) + x * math.sin(angle) - 0.5 * n * math.sin(angle) - 0.5 * m * math.cos(angle) + 0.5 * m
        return [X,Y]
    newrect=[]
    for i in range(4):
        print(rect[i])
        point=onepoint(rect[i][0]*m,rect[i][1]*n)
        newrect.append(point)
    #newrect.extend([1])
    print(newrect)
    return newrect
import numpy as np

points = np.array([[234,251], [253,255], [244,303], [225,299], [234,251], [242,292], [231,290],
 [236,258], [248,260], [242,292], [235,259], [248,264], [233,307], [220,302], [235,259],
 [240,297], [229,293], [243,255], [254,259], [240,297], [235,298], [224,291], [243,258]], np.int32)
#GetGoncateBounding(points)
CaluculateBoundingRect(points)
