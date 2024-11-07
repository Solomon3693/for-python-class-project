import cv2
import os
import time
import pickle
import numpy as np
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the labels and face data from pickle files
try:
    if os.path.exists('data/names.pkl') and os.path.exists('data/face_data.pkl'):
        with open('data/names.pkl', 'rb') as f:
            LABELS = pickle.load(f)
        with open('data/face_data.pkl', 'rb') as f:
            FACES = pickle.load(f)
    else:
        print("Error: Face data or names file not found.")
        exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Initialize KNN classifier with the loaded data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Start face detection and attendance marking
while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Get current timestamp for logging attendance
    ts = time.time()
    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

    # Set to track names already marked as present
    detected_names = set()

    # If faces are detected, mark them as Present
    if len(faces) > 0:
        print(f"Faces detected at {timestamp}. Marking Present:")

        for (x, y, w, h) in faces:
            # Crop and resize the detected face for prediction
            crop_img = frame[y:y + h, x:x + w]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            name = knn.predict(resized_img)[0]

            # If this name hasn't been marked as present already, mark it now
            if name not in detected_names:
                detected_names.add(name)
                print(f"{name} - Present")

            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)

    else:
        print(f"No faces detected at {timestamp}. Marking all as Absent:")
        for label in LABELS:
            print(f"{label} - Absent")

    # Display the frame with detected faces
    cv2.imshow("Attendance System", frame)

    k = cv2.waitKey(1)
    if k == ord('q'):  # If 'q' is pressed, quit the loop
        break

# Release resources
video.release()
cv2.destroyAllWindows()
