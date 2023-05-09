import cv2
import os

# Load the cascade
face_cascade = cv2.CascadeClassifier('C:/Users/Beulla/Documents/y3/faceDetection/haarcascade_frontalface_default.txt')

# Open the video stream
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening camera")
    exit()

# Define the codec and create VideoWriter object for output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Loop through each frame of the video stream
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Convert into grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Crop and save each face
        for i, (x, y, w, h) in enumerate(faces):
            # Crop the face
            face = frame[y:y+h, x:x+w]

            # Create a new file name
            filename = f'face_{i}.jpg'

            # Save the face
            cv2.imwrite(filename, face)

            print(f'Face {i} saved as {filename}')

        # Write the frame to the output video file
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
