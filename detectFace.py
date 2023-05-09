import cv2
import os

# Load the cascade
face_cascade = cv2.CascadeClassifier('C:/Users/Beulla/Documents/y3/faceDetection/haarcascade_frontalface_default.txt')

# Read the input image
img = cv2.imread('C:/Users/Beulla/Documents/y3/faceDetection/IMG_2963.jpg')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Crop and save each face
for i, (x, y, w, h) in enumerate(faces):
    # Crop the face
    face = img[y:y+h, x:x+w]

    # Create a new file name
    filename = f'face_{i}.jpg'

    # Save the face
    cv2.imwrite(filename, face)

    print(f'Face {i} saved as {filename}')

print('All faces saved successfully!')
