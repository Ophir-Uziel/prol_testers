from __future__ import division
import cv2
import face_recognition
import resize
from matplotlib.pyplot import *

import time
import sys
CONF_THRESHOLD = 0.7
DNN = "TF"
IMG_DIM = (300,300)


def choose_net():
    # OpenCV DNN supports 2 networks.
    # 1. FP16 version of the original caffe implementation ( 5.4 MB )
    # 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
    if DNN == "CAFFE":
        modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        modelFile = "dnn/opencv_face_detector_uint8.pb"
        configFile = "dnn/opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    return net


# the deafult dimensions were (300, 300)
def detectFaceOpenCVDnn(frame, dimensions):

    net = choose_net()

    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, dimensions, [104, 117, 123], False, False)

    net.setInput(blob)



    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_THRESHOLD:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes




def get_Bbox_from_Sbox(Sbox, Sresolution, Bresolution):
    ratioY = Bresolution[0]/Sresolution[0]
    ratioX = Bresolution[1]/Sresolution[1]
    x1 = int(ratioX * Sbox[0])
    y1 = int(ratioY * Sbox[1])
    x2 = int(ratioX * Sbox[2])
    y2 = int(ratioY * Sbox[3])
    Bbox = [x1, y1, x2, y2]
    return Bbox


def get_faces_from_image(image):
    shape = image.shape
    small_image = resize.resize(image, IMG_DIM)
    frame, boxes = detectFaceOpenCVDnn(small_image, IMG_DIM)
    faces = []
    for Sbox in boxes:
        Bbox = get_Bbox_from_Sbox(Sbox, IMG_DIM, shape)
        face = image[Bbox[1]:Bbox[3], Bbox[0]:Bbox[2]]
        if face.size > 0:
            faces.append(face)
    return faces


# path = '/mnt/d/FaceCompare/results_folderFN/compared_himself/a497/profile.png'
# real_image = face_recognition.load_image_file(path)
# faces = get_faces_from_image(real_image)
# path = '/mnt/d/FaceCompare/results_folderFN/compared_himself/a497/1profile.png'
# imsave(path, faces[0])
# print(faces[0].shape)
