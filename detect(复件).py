import ctypes 
import math
import random
import os
import cv2
import numpy as np
import time
import darknet.darknet as darknet

# 把坐标从yolo图片大小恢复到原图大小的坐标
# arg:x,y,w,h:是yolo输出的bbox变量；shape1是yolo模型输入图片的大小，shape2是原图大小
def xy_coord_warp(x, y, w, h, shape1, shape2):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    xmin = xmin / shape1[1] * shape2[1]
    xmax = xmax / shape1[1] * shape2[1]
    ymin = ymin / shape1[0] * shape2[0]
    ymax = ymax / shape1[0] * shape2[0]
    return int(xmin), int(ymin), int(xmax), int(ymax)

# 恢复原图坐标后，输出新的detection的信息包括了：bbox的坐标、种类、置信度。
# arg：detections是yolo输出的信息变量的list
def convertBack(detections, shape1, shape2):
    origin_detections = []
    for detection in detections:
        x, y, w, h = detection[2][0],\
                     detection[2][1],\
                     detection[2][2],\
                     detection[2][3]
        xmin, ymin, xmax, ymax = xy_coord_warp(float(x), float(y), float(w), float(h), shape1, shape2)
        origin_detections.append((detection[0], detection[1], [xmin, ymin, xmax, ymax]))
    return origin_detections

def cvDrawBoxes_on_origin_img(detections, positions, img):
    for i, detection in enumerate(detections):
        xmin, ymin, xmax, ymax = detection[2][0], detection[2][1], detection[2][2], detection[2][3]
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        position = positions[i]
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]:" + '{:.2f}'.format(position[2]),
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

# yolo模型参数初始化，调用一次后可执行detect。
def yolo_initialize():
    netMain = None
    metaMain = None
    altNames = None

    configPath = "./darknet/cfg/yolov4.cfg"
    weightPath = "./darknet/yolov4.weights"
    metaPath = "./darknet/cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    return netMain, metaMain

# yolo模型初始化
if __name__ == '__main__':
    print('loading...')
    netMain, metaMain = yolo_initialize()
    print('finish.')
