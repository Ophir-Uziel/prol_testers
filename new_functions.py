import FR_functions
import FN_functions

DIFFERENT = 0
FIRST_IMAGE_NO_FACES = 1
SECOND_IMAGE_NO_FACES = 2
SAME = 3


def compare_images(image1, image2, algorithm, face_locs1=None, face_locs2=None, faces1 = None, faces2 = None):
    if algorithm == 'FN':
        return FN_functions.compare_images_FN(image1, image2, known_faces1=faces1,
                                              known_faces2=faces2)
    elif algorithm == 'FR':
        return FR_functions.compare_images_FR(image1, image2, known_face_locations1=face_locs1,
                                              known_face_locations2=face_locs2)


def get_faces(img, algorithm, face_loc = None):
    if algorithm == 'FN':
        faces_dict = FN_functions.get_faces(img)
        return faces_dict
    elif algorithm == 'FR':
        return FR_functions.get_faces(img, face_loc)

def get_faces_loc(img, algorithm):
    if algorithm == 'FN':
        return None
    elif algorithm == 'FR':
        return FR_functions.get_faces_loc(img)