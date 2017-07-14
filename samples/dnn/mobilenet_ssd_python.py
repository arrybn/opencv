from __future__ import print_function
import numpy as np
import sys
sys.path.append('/home/arrybn/build/opencv/lib')
import cv2
from cv2 import dnn
import timeit

inWidth = 300
inHeight = 300
inScaleFactor = 0.007843
meanVal = 127.5
confThr = 0.2

classNames = ('background',
              'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor')

net = dnn.readNetFromCaffe('/hdd/caffe_models/MobileNet-SSD/MobileNetSSD_deploy_generated.prototxt',
                           '/hdd/caffe_models/MobileNet-SSD/MobileNetSSD_deploy_generated.caffemodel')

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    blob = dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), meanVal)
    net.setInput(blob)
    detections = net.forward()

    WHRatio = inWidth / float(inHeight)

    cols = frame.shape[1]
    rows = frame.shape[0]

    if cols / float(rows) > WHRatio:
        cropSize = (int(rows * WHRatio), rows)
    else:
        cropSize = (cols, int(cols / WHRatio))

    y1 = (rows - cropSize[1]) / 2
    y2 = y1 + cropSize[1]
    x1 = (cols - cropSize[0]) / 2
    x2 = x1 + cropSize[0]
    frame = frame[y1:y2, x1:x2]

    cols = frame.shape[1]
    rows = frame.shape[0]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confThr:
            class_id = int(detections[0, 0, i, 1])

            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop =   int(detections[0, 0, i, 5] * cols)
            yRightTop =   int(detections[0, 0, i, 6] * rows)

            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0))
            label = classNames[class_id] + ": " + str(confidence)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                             (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                      (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow("detections", frame)
    if cv2.waitKey(1) >= 0:
        break
