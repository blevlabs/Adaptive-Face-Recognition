import cv2
from face_profiler import recognize

def get_rgb_frame_and_return_face_data(video_capture):
    # Capture a frame from the video source (webcam or file)
    ret, frame = video_capture.read()

    if not ret:
        print("Error capturing frame")
        return None

    # Convert the captured frame from BGR to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get face data using the recognize function from face_profiler
    face_data = recognize(rgb_frame, array=True)

    return face_data