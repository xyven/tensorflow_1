######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import glob
import shutil
import cv2
import numpy as np
import math
import tensorflow as tf
import sys
import re
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw

#import self-define packages
from RadarPreprocessor import *

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from utils import visualization_utils_xu as vis_util_xu

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
#IMAGE_NAME = '20070725.031820.01.19.570.19.png'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH, 'radartemp', '*.png')

PATH_TO_GFOUTPUT = os.path.join(CWD_PATH, 'gustfrontoutput')


PATH_TO_R19 = os.path.join(CWD_PATH, 'radar19', '*.*')

# Number of classes the object detector can identify
NUM_CLASSES = 1

#load radar and transform to png-36
def main(rdfile):
    Make36AnglePNG(rdfile)
    pnglist = glob.glob(PATH_TO_IMAGE)
    for pngfile in pnglist:
        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier
        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # print(detection_scores)

        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image = cv2.imread(pngfile)
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        #print(num)

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=0.30,
            skip_labels=True)

        # All the results have been drawn on image. Now display the image.
        #cv2.imshow('Object detector', image)

        # Press any key to close the image
        #cv2.waitKey(0)
        #print(pngfile)
        #print(os.path.join(PATH_TO_GFOUTPUT, '%s'%(pngfile)))
        (pngfilepath, pngfilename) = os.path.split(pngfile)
        (filename, extension) = os.path.splitext(os.path.split(rdfile)[1])
        #创建文件夹
        if not os.path.exists(os.path.join(PATH_TO_GFOUTPUT, filename)):
            os.mkdir(os.path.join(PATH_TO_GFOUTPUT, filename))
        if num[0] > 0:
            cv2.imwrite(os.path.join(PATH_TO_GFOUTPUT, filename,'%s'%(pngfilename)), image)
        # Clean up
        cv2.destroyAllWindows()
    pass

#draw rotate boxes on an original image
def main2(rdfile):
    #rdfile='C:/tensorflow1/models/research/object_detection/radar19/20080602.033242.01.19.570'
    Make36AnglePNG(rdfile)
    pnglist = glob.glob(PATH_TO_IMAGE)
    oripng = 'C:/tensorflow1/models/research/object_detection/radartemp/temp.0.png'
    (rdfilepath, rdfilename) = os.path.split(rdfile)
    outpng = os.path.join('C:/tensorflow1/models/research/object_detection/radarrectcollection',
                          '%s.png' % (rdfilename))
    inclined_boxes = []
    for pngfile in pnglist:
        #if not os.path.exists(outpng):
        #    print('outpng is not exist!')
        #    outpng = oripng
        oriimage = cv2.imread(oripng)
        #print(outpng)
        #rotate angle from png filename for example temp.1.png means angle is 1
        (pngfilepath, pngfilename) = os.path.split(pngfile)
        rotate_ang = int(re.findall('\d+', pngfilename)[0])
        #if rotate_ang!=5:
        #    continue
        #(filename, extension) = os.path.splitext(os.path.split(oripng)[1])

        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier
        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # print(detection_scores)

        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image = cv2.imread(pngfile)
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        #print(num)
        #print(boxes)
        #对boxes进行旋转控制


        # Draw the results of the detection (aka 'visulaize the results')
        vis_util_xu.visualize_boxes_and_labels_on_image_array(
            oriimage,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=0.30,
            skip_labels=True,
            rotate_ang = rotate_ang)

        if num[0] > 0:
            cv2.imwrite(oripng, oriimage)
        # Clean up
        cv2.destroyAllWindows()
    shutil.copy(oripng, outpng)

#deal with the overlap boxes
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

def rotatecordiate(angle, rect):
    #print(angle)
    angle=angle*math.pi/180
    #print(angle)
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
        #print(rect[i])
        point=onepoint(rect[i][0]*m, rect[i][1]*n)
        newrect.append(point)
    #newrect.extend([1])
    #print(newrect)
    return newrect

def CaluculateminBoundingRect(obboxes):
    rect = cv2.minAreaRect(obboxes)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = np.append(box, box[0])
    box = box.reshape(-1, 2)
    return box

#####
#旋转图片用硬盘做缓存，输出只输出最终结果
#####
def main3(rdfile):
    #rdfile='C:/tensorflow1/models/research/object_detection/radar19/20070802.083232.01.19.570'
    reader19 = Make36AnglePNG(rdfile)

    pnglist = glob.glob(PATH_TO_IMAGE)
    oripng = 'C:/tensorflow1/models/research/object_detection/radartemp/temp.0.png'
    (rdfilepath, rdfilename) = os.path.split(rdfile)
    outpng = os.path.join('C:/tensorflow1/models/research/object_detection/radarrectcollection', '%s.png' % (rdfilename))
    #outpng = oripng
    inclined_boxes = []
    label_boxes = []
    overlap_n_boxes = []
    for pngfile in pnglist:
        print(pngfile)
        (pngfilepath, pngfilename) = os.path.split(pngfile)
        rotate_ang = int(re.findall('\d+', pngfilename)[0])
        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier
        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # print(detection_scores)

        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image = cv2.imread(pngfile)
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        #print(num)
        #print(boxes)
        # 对boxes进行旋转控制
        for n in range(int(num[0])):
            ymin, xmin, ymax, xmax = boxes[0,n,:]
            orect = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            newrect = rotatecordiate(rotate_ang * 10, orect)
            absrect = np.array(newrect) * 460
            # print(rect)
            close_rect = list(absrect)
            close_rect.append(absrect[0])
            np_close_rect = np.array(close_rect, np.int32)
            #tuple_rect = list(map(tuple, close_rect))
            #print(np_close_rect)
            inclined_boxes.append(np_close_rect)
            label_boxes.append(0)
            overlap_n_boxes.append(0)
            #print(np_close_rect)
    #print(len(inclined_boxes))
    cur_box_idx = 0
    cur_lable = 1
    for box_o in inclined_boxes:
        if label_boxes[cur_box_idx] == 0:
            label_boxes[cur_box_idx] = cur_lable
            cur_lable = cur_lable + 1
        noverlap=0
        for i in range(len(inclined_boxes)):
            box_r = inclined_boxes[i]
            or_area, and_area, IOU = CalculateIOU(box_o, box_r)
            if IOU>0.99:
                continue
            if IOU>0.10:
                noverlap = noverlap+1
                label_boxes[i] = label_boxes[cur_box_idx]
        overlap_n_boxes[cur_box_idx] = noverlap
        cur_box_idx = cur_box_idx+1
    #draw image
    oriimage = DrawRGBRadarImage(reader19)
    draw = ImageDraw.Draw(oriimage)
    classfied_boxes_points = [[]]*cur_lable
    for idx_boxes in range(len(inclined_boxes)):
        if overlap_n_boxes[idx_boxes] == 0:
            continue
        if len(classfied_boxes_points[label_boxes[idx_boxes]-1]) == 0:
            classfied_boxes_points[label_boxes[idx_boxes] - 1] = inclined_boxes[idx_boxes]
        else:
            classfied_boxes_points[label_boxes[idx_boxes] - 1] = np.concatenate((classfied_boxes_points[label_boxes[idx_boxes]-1], inclined_boxes[idx_boxes]), axis=0)
        #多边形逼近
        #draw = ImageDraw.Draw(image_pil)
        #tuple_box = list(map(tuple, inclined_boxes[idx_boxes]))
        #draw.line(tuple_box, width=2, fill='AliceBlue')
    #print(classfied_boxes_points)
    #np.copyto(oriimage, np.array(image_pil))
    for cbps in classfied_boxes_points:
        acbps = np.array(cbps, np.int32)
        if len(acbps) == 0:
            continue
        minbound = CaluculateminBoundingRect(acbps)
        #cv2.drawContours(oriimage, [minbound], 0, (255, 255, 0), 2)
        draw.line(list(map(tuple, minbound)), fill=(255, 255, 0), width=3)
    #cv2.imwrite(oripng, oriimage)
    oriimage.save(oripng)
    shutil.copy(oripng, outpng)
    pass

#####
#优化速度
#####
def main4(rdfile):
    #rdfile='C:/tensorflow1/models/research/object_detection/radar19/20070802.083232.01.19.570'
    reader19, imgs, angles = Make36AngleImgArray(rdfile)
    pnglist = glob.glob(PATH_TO_IMAGE)
    oripng = 'C:/tensorflow1/models/research/object_detection/radartemp/temp.0.png'
    (rdfilepath, rdfilename) = os.path.split(rdfile)
    outpng = os.path.join('C:/tensorflow1/models/research/object_detection/radarrectcollection', '%s.png' % (rdfilename))
    inclined_boxes = []
    label_boxes = []
    overlap_n_boxes = []
    for img_i in range(len(imgs)):
        img = imgs[img_i]
        rotate_ang = angles[img_i]
        #(pngfilepath, pngfilename) = os.path.split(pngfile)
        #rotate_ang = int(re.findall('\d+', pngfilename)[0])
        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=detection_graph)
        # Define input and output tensors (i.e. data) for the object detection classifier
        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # print(detection_scores)
        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image = np.array(img)
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        # 对boxes进行旋转控制
        for n in range(int(num[0])):
            ymin, xmin, ymax, xmax = boxes[0,n,:]
            orect = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            newrect = rotatecordiate(rotate_ang * 10, orect)
            absrect = np.array(newrect) * 460
            # print(rect)
            close_rect = list(absrect)
            close_rect.append(absrect[0])
            np_close_rect = np.array(close_rect, np.int32)
            #tuple_rect = list(map(tuple, close_rect))
            #print(np_close_rect)
            inclined_boxes.append(np_close_rect)
            label_boxes.append(0)
            overlap_n_boxes.append(0)
            #print(np_close_rect)
    #print(len(inclined_boxes))
    cur_box_idx = 0
    cur_lable = 1
    for box_o in inclined_boxes:
        if label_boxes[cur_box_idx] == 0:
            label_boxes[cur_box_idx] = cur_lable
            cur_lable = cur_lable + 1
        noverlap=0
        for i in range(len(inclined_boxes)):
            box_r = inclined_boxes[i]
            or_area, and_area, IOU = CalculateIOU(box_o, box_r)
            if IOU>0.99:
                continue
            if IOU>0.10:
                noverlap = noverlap+1
                label_boxes[i] = label_boxes[cur_box_idx]
        overlap_n_boxes[cur_box_idx] = noverlap
        cur_box_idx = cur_box_idx+1
    #draw image
    oriimage = DrawRGBRadarImage(reader19)
    draw = ImageDraw.Draw(oriimage)
    classfied_boxes_points = [[]]*cur_lable
    for idx_boxes in range(len(inclined_boxes)):
        if overlap_n_boxes[idx_boxes] == 0:
            continue
        if len(classfied_boxes_points[label_boxes[idx_boxes]-1]) == 0:
            classfied_boxes_points[label_boxes[idx_boxes] - 1] = inclined_boxes[idx_boxes]
        else:
            classfied_boxes_points[label_boxes[idx_boxes] - 1] = np.concatenate((classfied_boxes_points[label_boxes[idx_boxes]-1], inclined_boxes[idx_boxes]), axis=0)
    for cbps in classfied_boxes_points:
        acbps = np.array(cbps, np.int32)
        if len(acbps) == 0:
            continue
        minbound = CaluculateminBoundingRect(acbps)
        #cv2.drawContours(oriimage, [minbound], 0, (255, 255, 0), 2)
        draw.line(list(map(tuple, minbound)), fill=(255, 255, 0), width=3)
    #cv2.imwrite(oripng, oriimage)
    oriimage.save(oripng)
    shutil.copy(oripng, outpng)
    pass

def DrawRGBRadarImage(reader19):
    # 输出色块图
    idcrdata = np.array(list(reader19.Product.iDcrData), dtype=np.uint8)
    array_idd = idcrdata.reshape((460, 460))
    colorlist = ((0, 0, 0, 255), (0, 172, 164, 255), (192, 192, 254, 255), (122, 114, 238, 255), (30, 38, 208, 255),
                 (166, 252, 168, 255), (0, 234, 0, 255),
                 (16, 146, 26, 255), (252, 244, 100, 255), (200, 200, 2, 255), (140, 140, 0, 255), (254, 172, 172, 255),
                 (254, 100, 92, 255), (238, 2, 48, 255),
                 (212, 142, 254, 255), (170, 36, 250, 255))
    im_n = Image.new("RGB", (460, 460), "white")
    for x in range(460):
        for y in range(460):
            im_n.putpixel((x, y), colorlist[array_idd[y, x]])
    #imgfile_a3 = os.path.join(r'radartemp', r'temp.xxxxxy.png')
    #im_n.save(imgfile_a3)
    return im_n
if __name__ == '__main__':
    #main3()
    rdlist = glob.glob(PATH_TO_R19)
    for rdfile in rdlist:
        main3(rdfile)

