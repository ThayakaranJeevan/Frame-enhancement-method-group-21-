import cv2
import os

video_path = "road_1080p.mp4"

original_folder = "original_frames"
denoised_folder ="denoised_frames"

os.makedirs(original_folder, exist_ok=True)
os.makedirs(denoised_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 1
f=0

while True:
    ret, frame = cap.read()

    if not ret:
        break 
    if frame_count % 41 == 0:
        frame_name = f"{original_folder}/original_frame_{frame_count:05d}.jpg"
        cv2.imwrite(frame_name, frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
        median = cv2.medianBlur(gaussian, 3)

        frame_name = f"{denoised_folder}/denoised_frame_{frame_count:05d}.jpg"
        cv2.imwrite(frame_name, median)
        
        f += 1

    frame_count += 1

cap.release()

print(f"Done! {f} frames saved.")
