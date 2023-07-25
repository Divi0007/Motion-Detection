import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

video_path = "/Users/divyanshyadav/Downloads/Untitled.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error opening video file:", video_path)
    exit()

# Get the original width and height of the video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a background subtractor
mog = cv2.createBackgroundSubtractorMOG2()

# Initialize variables for machine running state
last_motion_time = time.time()
machine_running = True

# Create a window to display the output
cv2.namedWindow('Motion Detection', cv2.WINDOW_NORMAL)

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame was not successfully read, exit the loop
    if not ret:
        break
    frame = cv2.resize(frame, (1080, 720))
    roi = frame[500:620]
    # Convert the frame to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction
    fgmask = mog.apply(gray)

    # Perform morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    # Find contours
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if there are any significant motions
    motion_detected = False
    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) < 1000:
            continue

        # Draw bounding box around contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Motion detected
        motion_detected = True

    # Update machine running state based on motion detection
    if motion_detected:
        last_motion_time = time.time()
        machine_running = True
    elif time.time() - last_motion_time > 5:
        machine_running = False

    # Add machine running label on top right of the frame
    label = "Machine Running" if machine_running else "Machine Not Running"
    cv2.putText(frame, label, (450,420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Resize the frame to match the original dimensions
    frame = cv2.resize(frame, (width, height))

    # Display the frame with bounding rectangles and machine running label
    cv2.imshow('Motion Detection', frame)

    # Check if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()