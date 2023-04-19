import numpy as np
from facenet_models import FacenetModel
import cv2
import skimage.io as io

facenet = FacenetModel()


def img_to_array(image_path):
    """ Converts image from image path into numpy array
    Parameters:
    image_path: string

    Returns:
    numpy array
    """
    image = io.imread(str(image_path))
    if image.shape[-1] == 4:
        image = image[..., :-1]  # png -> RGB

    return image


def userinput(camera=False, image_directory="", array=None):
    if camera:
        cam = cv2.VideoCapture(0)
        ret, image = cam.read()
        cam.release()
    elif array is not None:
        image = array
    else:
        assert image_directory != "", "Please enter a valid image directory"
        image = img_to_array(image_directory)

    boxes, probabilities, landmarks = facenet.detect(image)
    if boxes is None:
        return None, None

    boxes = np.array([box for i, box in enumerate(boxes) if probabilities[i] > 0.9])
    probabilities = np.array([prob for prob in probabilities if prob > 0.9])

    return boxes, facenet.compute_descriptors(image, boxes)


class Profiling:
    def __init__(self, name):
        self.name = name
        self.array_of_descriptors = np.array([])
        self.average_descriptor = None

    @property
    def parameters(self):
        return self.name, self.array_of_descriptors, self.average_descriptor

    def add_descriptor_vectors(self, descriptors):
        if self.array_of_descriptors.size == 0:
            self.array_of_descriptors = descriptors.reshape((1, 512))
        else:
            self.array_of_descriptors = np.vstack((self.array_of_descriptors, descriptors))

        self.average_descriptor = np.average(self.array_of_descriptors, axis=0)