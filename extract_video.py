import cv2
import os

video_path = "./assets/calibration_video.mp4"
output_dir = "./assets/calibration"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

os.makedirs(output_dir, exist_ok=True)

frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Save the frame with the original resolution.
    if frame_index % 5 == 0:
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_index:04d}.png"), frame)
    frame_index += 1

cap.release()
print(f"Saved {frame_index} frames to '{output_dir}'.")