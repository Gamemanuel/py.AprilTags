import cv2
import numpy as np
from robotpy_apriltag import AprilTagDetector, AprilTagPoseEstimator
import wpimath.units as units

# Initialize AprilTag detector and enable tag family
detector = AprilTagDetector()
detector.addFamily('tag36h11')

# Define the camera parameters (intrinsics) - these values should be calibrated for your specific camera
camera_matrix = np.array([[600.0, 0.0, 320.0],
                          [0.0, 600.0, 240.0],
                          [0.0, 0.0, 1.0]])
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion for simplicity

# Define the size of the AprilTag in meters
tag_size = 0.165  # Example: 16.5 cm

# Convert the tag size to the required unit type
tag_size_meters = units.meters(tag_size)

# Create AprilTagPoseEstimator configuration
config = AprilTagPoseEstimator.Config(tag_size_meters, camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2])
pose_estimator = AprilTagPoseEstimator(config)

# Capture video from the webcam
cap = cv2.VideoCapture(0) #defines what camera index that you want

if not cap.isOpened():
    print("Fatal Error: Could not open the camera code:E1")
    exit() # kills the program because it is a fatal error

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Fatal Error: Could not camera input code:E2")
        break #kills the loop and attempts to try again

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # Detect AprilTags
    detections = detector.detect(gray_frame)

    # Draw bounding boxes, display ID, and calculate pose for detected tags
    for detection in detections:
        # Define the buffer for corners as a tuple with eight float values
        cornersBuf = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        corners = detection.getCorners(cornersBuf)
        corners = np.int32(np.array(corners).reshape((4, 2)))

        # Draw bounding box
        cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw center point
        center = (int(detection.getCenter()[0]), int(detection.getCenter()[1]))
        cv2.circle(frame, center, radius=5, color=(0, 0, 255), thickness=-1)

        # Display ID
        tag_id = detection.getId()
        cv2.putText(frame, f"ID: {tag_id}", (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Estimate pose
        pose = pose_estimator.estimate(detection)

        # Get translation and rotation from the pose
        translation = pose.translation()
        rotation = pose.rotation()

        # Display translation and rotation near the tag
        translation_str = f"T: {translation.x:.2f}, {translation.y:.2f}, {translation.z:.2f}"
        rotation_str = f"R: {rotation.x:.2f}, {rotation.y:.2f}, {rotation.z:.2f}"
        cv2.putText(frame, translation_str, (center[0] - 10, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, rotation_str, (center[0] - 10, center[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('AprilTag Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
