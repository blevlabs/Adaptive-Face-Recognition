import os
import string
import random

from face_database import Database
from profiling import userinput


def recognize(imgfp=None, databasefp="face_profiles.pkl",
              extractKnown=False, camera=False, array=False):
    if array:
        img = imgfp
        uituple = userinput(array=imgfp)
    else:
        uituple = userinput(camera=camera, image_directory=imgfp)

    if uituple[0] is None:
        return None

    dtb = Database(databasefp)
    final_detections = {}

    for x, i in zip(uituple[0], uituple[1]):
        name = dtb.query(i)
        newX = [float(y) for y in x]
        final_detections[name] = {"data": list(newX), "emote": "neutral"}

    known_face_names = list(set(final_detections.keys()))
    if "Unknown" in known_face_names:
        known_face_names.remove("Unknown")

    for name in known_face_names:
        random_filename = name + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
        emote = dtb.extract_face_and_update_profile(img, final_detections[name], name=name)
        final_detections[name]["emote"] = emote

    return final_detections