# Adaptive Face Recognition and Emotion Detection

This project provides a comprehensive solution for face recognition and emotion detection in images. It consists of several modules that handle different aspects of the process, such as profiling, database management, and image processing.

## Project Structure

The project is organized into the following modules:

- `emotion.py`: Contains the `EmotionRecognition` class, which is used for detecting emotions in facial images.
- `face_database.py`: Contains the `Database` class for managing a face recognition database stored in a pickled file.
- `profiling.py`: Contains the `Profiling` class for managing individual profiles, including name and average descriptor, and the `userinput` function for capturing images and computing face descriptors.
- `face_profiler.py`: Contains the main `recognize` function that integrates all other modules to recognize faces and detect emotions in a given image.
- `run.py`: Contains a demo function `get_rgb_frame_and_return_face_data` that utilizes an OpenCV VideoCapture object to show the use of the `recognize` function above
## Database Initialization

Before using the face recognition system, you need to initialize an empty face recognition database by calling the `database_initialization` function from the `face_database` module:

```python
from face_database import database_initialization

database_initialization(dir="path/to/database.pkl")
```

This will create a new pickled file at the specified path to store the face recognition database.

## Modules Overview

### emotion.py

The `EmotionRecognition` class in this module is used for detecting emotions in facial images. It takes an image as input and returns the detected emotion as output. To use this class, you need to create an instance of it and call the `run` method with an image:

```python
from emotion import EmotionRecognition
from PIL import Image

emotion_recognizer = EmotionRecognition()

image = Image.open("path/to/image.jpg")
detected_emotion = emotion_recognizer.run(image)
```

### face_database.py

This module contains the `Database` class, which is responsible for managing the face recognition database. It provides methods to add new profiles, update existing profiles with new descriptors, and query the database to identify faces. To use this class, you need to create an instance of it and call the appropriate methods:

```python
from face_database import Database

db = Database(database_directory="path/to/database.pkl")
db.creation(directory="path/to/image/directory")
```

### profiling.py

The `Profiling` class in this module manages individual profiles for face recognition, including the name and average descriptor of each person. The class also provides a method to add new descriptor vectors to the profile and update the average descriptor accordingly.

The `userinput` function captures images from a camera or an image directory, detects faces, and computes descriptors using the `FacenetModel` class.

To use the `Profiling` class and `userinput` function, follow these steps:

```python
from profiling import Profiling, userinput

profile = Profiling(name="John Doe")
boxes, descriptors = userinput(camera=True)
profile.add_descriptor_vectors(descriptors)
```

### face_profiler.py

This module contains the main `recognize` function that integrates all other modules to recognize faces and detect emotions in a given image. The function accepts an image file path or a NumPy array of the image and returns a dictionary containing the recognized face names, their corresponding bounding box coordinates, and detected emotions.

To use the `recognize` function, follow these steps:

```python
from recognize import recognize

result = recognize(imgfp="path/to/image.jpg")
```

## Usage

To use the Adaptive Face Recognition and Emotion Detection system, follow these steps:

1. Initialize the face recognition database.
2. Create instances of the `Database`, `EmotionRecognition`, and `Profiling` classes as needed.
3. Use the `recognize` function to recognize faces and detect emotions in a given image. See `run.py` for an example usage of this function
4. Process the results to visualize or update the face recognition database.

For more detailed usage instructions, refer to the documentation for each module and function.