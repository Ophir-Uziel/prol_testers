import tensorflow as tf
import numpy as np
from facematch import facenet
from facematch.align import detect_face
import cv2
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--img1", type = str, required=True)
# parser.add_argument("--img2", type = str, required=True)
# args = parser.parse_args()

# some constants kept as default from facenet
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
MARGIN = 44
INPUT_IMAGE_SIZE = 160

DIFFERENT = 0
FIRST_IMAGE_NO_FACES = 1
SECOND_IMAGE_NO_FACES = 2
SAME = 3

sess = tf.Session()

# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = detect_face.create_mtcnn(sess, 'facematch/align')

# read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
#facenet.load_model("facematch/20170512-110547/20170512-110547.pb")
#facenet.load_model("facematch/20180408-102900/20180408-102900.pb")
facenet.load_model("facematch/20180402-114759/20180402-114759.pb")
# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]


def get_embedding(resized):
    reshaped = resized.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding


def get_faces(img):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - MARGIN / 2, 0)
                bb[1] = np.maximum(det[1] - MARGIN / 2, 0)
                bb[2] = np.minimum(det[2] + MARGIN / 2, img_size[1])
                bb[3] = np.minimum(det[3] + MARGIN / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)
                faces.append(
                    {'face': resized, 'rect': [bb[0], bb[1], bb[2], bb[3]], 'embedding': get_embedding(prewhitened)})
    return faces


def compare_images_FN(img1, img2, threshold=1.0, known_faces1=None, known_faces2=None):
    if known_faces1 == None:
        image1_faces = get_faces(img1)
    else:
        image1_faces = known_faces1
    if known_faces2 == None:
        image2_faces = get_faces(img2)
    else:
        image2_faces = known_faces2

    if len(image1_faces) == 0:
        return FIRST_IMAGE_NO_FACES, None, None
    elif len(image2_faces) == 0:
        return SECOND_IMAGE_NO_FACES, None, None
    for i in range(len(image1_faces)):
        for j in range(len(image2_faces)):
            face1 = image1_faces[i]
            face2 = image2_faces[j]
            dist = np.sqrt(np.sum(np.square(np.subtract(face1['embedding'], face2['embedding']))))
            if dist < threshold:
                return SAME, i, j
    return DIFFERENT, None, None

# gets 2 face images and returns the distance between them
def dis_of_2_faces_FN(face1, face2):
    resized1 = cv2.resize(face1, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    prewhitened1 = facenet.prewhiten(resized1)
    resized2 = cv2.resize(face2, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    prewhitened2 = facenet.prewhiten(resized2)
    embedding1 = get_embedding(prewhitened1)
    embedding2 = get_embedding(prewhitened2)
    dist = np.sqrt(np.sum(np.square(np.subtract(embedding1, embedding2))))
    return dist

# get 2 faces lists, one for each image and returns the minimum distance between 2 faces
def dis_of_many_faces_FN(faces1, faces2):

    if len(faces1) == 0:
        return None, None, None
    elif len(faces2) == 0:
        return None, None, None

    enter = True
    min_i = 0
    min_j = 0

    for i in range(len(faces1)):
        for j in range(len(faces2)):
            face1 = faces1[i]
            face2 = faces2[j]
            if face2.size == 0:
                print("sss")
            if enter:
                enter = False
                min_dist = dis_of_2_faces_FN(face1, face2)
            else:
                curr_dist = dis_of_2_faces_FN(face1, face2)
                min_dist = min(min_dist, curr_dist)
                min_i = i
                min_j = j
    return min_dist, min_i, min_j

# gets 2 faces lists and threshold and returns whenever the minimum
def compare_faces_FN(faces1, faces2, threshold=1.1):

    if len(faces1) == 0:
        return FIRST_IMAGE_NO_FACES, None, None
    elif len(faces2) == 0:
        return SECOND_IMAGE_NO_FACES, None, None

    dist, min_i, min_j = dis_of_many_faces_FN(faces1, faces2)

    if dist <= threshold:
        return SAME, min_i, min_j
    return DIFFERENT, None, None
