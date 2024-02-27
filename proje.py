import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pylot as plt

# Step 1: Create a Poselandmarker object

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing.utils
detector = mp_pose.Pose(
    static_image_mod=True,
    model_complexity=2,
    enable_segmentation=True)

# Step 2: Load the input image.
image = cv2.imread(r"C:\Users\alpar\OneDrive\Desktop\Trainer_ai_proje\virat.png")

# Step 3: Convert the image to RGB format and process it
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = detector.process(image_rgb)
print(results.pose_landmarks.landmark(0))
# Step 4: Process the detection result and visualize it
annotated_image = image.copy()
mp_drawing_draw_landmarks(
    annotated_image,
    results_pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
    connection_drawing_spec=mp_drawing_DrawingSpec(color=(0, 255, 0), thickness=2))
print(len(results.pose_landmarks.landmarks))
for pose in results.pose_landmark.landmark:

    print(pose.x, pose.y, pose.z)
    