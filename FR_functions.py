import cv2
import requests
import numpy as np
import os
import face_recognition

DIFFERENT = 0
FIRST_IMAGE_NO_FACES = 1
SECOND_IMAGE_NO_FACES = 2
SAME = 3


def encode_image_faces(image, known_face_locations=None):
    return face_recognition.face_encodings(image, known_face_locations)


def compare_between_encoders(image1_encodings, image2_encodings, tolerance):
    results = False
    for i in range(len(image1_encodings)):
        for j in range(len(image2_encodings)):
            image1_encoding = image1_encodings[i]
            image2_encoding = image2_encodings[j]
            if face_recognition.compare_faces([image1_encoding], image2_encoding, tolerance)[0]:
                return SAME,i,j
    return DIFFERENT, None, None


def compare_images_FR(image1, image2, tolerance=0.6, known_face_locations1=None, known_face_locations2=None):
    if known_face_locations1 == None:
        image1_encodings = encode_image_faces(image1, get_faces_loc(image1))
    else:
        image1_encodings = encode_image_faces(image1, known_face_locations1)
    if known_face_locations2 == None:
        image2_encodings = encode_image_faces(image2, get_faces_loc(image2))
    else:
        image2_encodings = encode_image_faces(image2, known_face_locations2)

    if len(image1_encodings) == 0:
        return FIRST_IMAGE_NO_FACES, None, None
    if len(image2_encodings) == 0:
        return SECOND_IMAGE_NO_FACES, None, None
    return compare_between_encoders(image1_encodings, image2_encodings, tolerance)


def get_faces_loc(img, model='cnn'):
    return face_recognition.face_locations(img, model=model)


def get_faces(img, face_locs):
    if len(face_locs) == 0:
        return face_locs
    faces = []
    for loc in face_locs:
        top = loc[0]
        right = loc[1]
        bottom = loc[2]
        left = loc[3]
        faces.append(img[top:bottom, left:right])
    return faces
