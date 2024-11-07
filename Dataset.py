import cv2
import os
import numpy as np
import pickle

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# Initialize face data and labels
face_data = []
labels = []

# Loop to collect data for multiple people
while True:
    name = input("Enter your name (or type 'done' to stop): ")

    if name.lower() == 'done':
        break

    print(f"Collecting face samples for {name}... Press ESC to stop.")
    
    # Collect 100 samples for the current person
    sample_count = 0
    while sample_count < 100:
        ret, frame = video.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w]
            resized_img = cv2.resize(crop_img, (50, 50))
            face_data.append(resized_img.flatten())  # Flatten image for storing
            labels.append(name)  # Save the name for the face

            sample_count += 1

            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with rectangles around faces
        cv2.imshow("Collecting Face Data", frame)

        # Check for ESC key to stop the process
        if cv2.waitKey(1) == 27:  # Press ESC to break
            break

    print(f"Collected {sample_count} samples for {name}.")

# Release resources
video.release()
cv2.destroyAllWindows()

# Convert face data to numpy array and reshape
face_data = np.array(face_data)
face_data = face_data.reshape(face_data.shape[0], -1)

# Save the collected face data and names
with open('data/names.pkl', 'wb') as f:
    pickle.dump(labels, f)  # Save the names list for each face collected
with open('data/face_data.pkl', 'wb') as f:
    pickle.dump(face_data, f)

print(f"Saved {face_data.shape[0]} face samples for {len(set(labels))} people.")
