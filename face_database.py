import pickle
import os
import traceback

import cv2

os.path.join(os.path.dirname(__file__))
from profiling import Profiling, userinput
import numpy as np
from facenet_models import FacenetModel
from PIL import Image

facenet = FacenetModel()
from emotion import EmotionRecognition

emotion = EmotionRecognition()


def cos_dist(vector_a, vector_b):
    """ Calculates the cosine distance between two vectors
    Parameters:
        vector_a: numpy array of shape (N,)
        vector_b: numpy array of shape (N,)
    Return:
        cos_dist: float
    """
    a_magn = np.sqrt(np.sum(vector_a ** 2))
    b_magn = np.sqrt(np.sum(vector_b ** 2))
    cos_dist = 1 - np.dot(vector_a, vector_b) / (a_magn * b_magn)
    return cos_dist


def database_initialization(dir=""):
    with open(dir, "wb") as pklf:
        pickle.dump({}, pklf)
        pklf.close()


class Database:
    """
    Class to store profiles and methods to interact/query from those profiles
    """

    def __init__(self, database_directory):
        '''
        Initializes class for the database management system.
        Database holds a tuple of data from the Profile class of an individual.
        key = self.name
        value = (self.name,self.list_of_descriptors,self.average_descriptor)
        '''
        self.dir = database_directory
        self.open_database()

    def open_database(self):
        with open(self.dir, "rb") as pklf:
            self.database = pickle.load(pklf)
            pklf.close()

    def save_database(self):
        with open(self.dir, "wb") as pklf:
            pickle.dump(self.database, pklf)
            pklf.close()

    def creation(self, directory):
        """
        This function takes the directory that contains subdirectories of individuals to add to the face recognition database.
        Format:
        key = name
        value = Profile() class user instance. Profile(name).parameters returns (self.name, self.list_of_descriptors, self.average_descriptor)
        """
        for _, dirs, files in os.walk(directory, topdown=True):
            for f in files:
                name = f.replace("_", " ")
                name = name.split(".")[0]
                user_input_tuple = userinput(camera=False, image_directory=directory + "/" + f)
                self.add_descriptors(name=name, descriptors=user_input_tuple[-1])
        return

    def add_descriptors(self, name, descriptors):
        """
        Can be called to either add more descriptors to an existing profile or to add a new profile to the database
        If a new profile is added to the database, it adds the profile as the value to a dictionary with key of their name
        """
        if name not in self.database.keys():
            self.database[name] = Profiling(name)
        else:
            self.database[name].add_descriptor_vectors(descriptors)
        with open(self.dir, mode="wb") as file:
            pickle.dump(self.database, file)
            file.close()

    def extract_face_and_update_profile(self, img, coordinates, name=""):
        # x1, y1, x2, y2 = coordinates
        coordinates = [int(x) for x in coordinates["data"]]
        img = img[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
        try:
            img = Image.fromarray(img)
            emote = emotion.run(img)
            print(emote)
        except Exception as e:
            print(traceback.format_exc())
            emote = "Unknown"
        try:
            uituple = userinput(array=img)
            descriptors = uituple[-1]
            self.add_descriptors(name=name, descriptors=descriptors)
        except Exception as e:
            pass
        return emote

    def query(self, descriptors):
        """
        Takes in a picture that is not in the database and uses existing database profiles to identify people in the picture
        Parameters:
            descriptors: numpy array of descriptor vectors that is associated with the picture we are querying for face identifications
        Return:
            Name of the person identified or "Unknown" if there is no confident match
        """
        distances = []

        # For each profile in the database, the cosine distance is computed between the querying descriptor vectors and the average descriptors for each profile
        for name in self.database:
            profile = self.database[name]
            avg_descriptor = profile.average_descriptor
            dist = cos_dist(descriptors, avg_descriptor)
            distances.append(dist)

        # Locates the lowest cosine distance as the profile the image is most similar to
        distances = np.array(distances)
        min_dist_idx = np.argmin(distances)
        if distances[min_dist_idx] < 0.5:  # placeholder threshold, tb optimized later
            self.add_descriptors(name, descriptors)
            return list(self.database.keys())[min_dist_idx]
        else:
            return "Unknown"

    def draw_name_box(self, name, coords, fp):
        coords = [int(x) for x in coords]
        img = cv2.imread(fp)
        color = (255, 0, 0)
        if name == "Unknown":
            color = (0, 0, 255)
        cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
        cv2.putText(img, name, (coords[0], coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        cv2.imwrite(fp, img)
