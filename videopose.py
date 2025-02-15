import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model for person detection
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

#BY LIVE CAMERA
#video_path = 0

# by video Path to the input video
video_path = r"C:\Users\Rahul\Videos\Screen Recordings\Screen Recording 2025-02-15 092439.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer to save processed frames
output_path = "output_pose_estimation.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video processing completed.")
            break

        # Run YOLOv8 for person detection
        results = model(frame, verbose=False)
        persons = []

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Class '0' corresponds to 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    persons.append((x1, y1, x2, y2))

        # Process each detected person with MediaPipe Pose
        for (x1, y1, x2, y2) in persons:
            person_crop = frame[y1:y2, x1:x2]  # Crop person from the frame
            if person_crop.size == 0:
                continue  # Skip empty crops

            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            results = pose.process(person_rgb)

            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    h, w, _ = person_crop.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x1 + cx, y1 + cy), 2, (0, 255, 0), -1)  # Draw landmarks

                # Draw pose connections
                mp_drawing.draw_landmarks(
                    frame[y1:y2, x1:x2],
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )

        # Write processed frame to output video
        out.write(frame)

        # Display output
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Multi-Person Pose Estimation", display_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved at: {output_path}")
