import cv2


def resize(image, dimensions):
    max_height, max_width = dimensions
    current_size = image.shape
    if current_size[0] <= max_height and current_size[1] <= max_width:
        return image
    factor = min(max_height / current_size[0], max_width / current_size[1])
    a = cv2.resize(image, (0, 0), fx=factor, fy=factor)
    return a
