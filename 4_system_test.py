'''

import cv2
import dlib
import time
import matplotlib.pyplot as plt

im = cv2.imread('facematch/images/amitab_old.jpg', 0)

CONF_THRESHOLD = 0.6

im = cv2.resize(im,(300, 300), interpolation = cv2.INTER_CUBIC)

frameHeight = im.shape[0]
frameWidth = im.shape[1]

start_time = time.time()
#Haar Cascade
# faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
# faces = faceCascade.detectMultiScale(im)
# for face in faces:
#     x1, y1, w, h = face
#     x2 = x1 + w
#     y2 = y1 + h
#
# cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),3)

time_1 = time.time()
#Dnn
# DNN = "TF"
# if DNN == "CAFFE":
#     modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
#     configFile = "deploy.prototxt"
#     net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
# else:
#     modelFile = "opencv_face_detector_uint8.pb"
#     configFile = "opencv_face_detector.pbtxt"
#     net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
#
# blob = cv2.dnn.blobFromImage(im, 1.0, (300, 300), [104, 117, 123], False, False)
#
# net.setInput(blob)
# detections = net.forward()
# bboxes = []
# for i in range(detections.shape[2]):
#     confidence = detections[0, 0, i, 2]
#     if confidence > CONF_THRESHOLD:
#         x1 = int(detections[0, 0, i, 3] * frameWidth)
#         y1 = int(detections[0, 0, i, 4] * frameHeight)
#         x2 = int(detections[0, 0, i, 5] * frameWidth)
#         y2 = int(detections[0, 0, i, 6] * frameHeight)
#
# cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),3)


time_2 = time.time()
#Hog
hogFaceDetector = dlib.get_frontal_face_detector()
faceRects = hogFaceDetector(im, 0)
for faceRect in faceRects:
    x1 = faceRect.left()
    y1 = faceRect.top()
    x2 = faceRect.right()
    y2 = faceRect.bottom()

cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),3)


time_3 = time.time()
#Cnn
dnnFaceDetector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
faceRects = dnnFaceDetector(im, 0)
for faceRect in faceRects:
    x1 = faceRect.rect.left()
    y1 = faceRect.rect.top()
    x2 = faceRect.rect.right()
    y2 = faceRect.rect.bottom()

cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),3)

time_end = time.time()

print("Haar: ", time_1 - start_time)
print("Haar: ", time_2 - time_1)
print("Haar: ", time_3 - time_2)
print("CNN: ", time_end - time_3)


'''
