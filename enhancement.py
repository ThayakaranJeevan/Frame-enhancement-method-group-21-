import cv2
import os
import numpy as np

video_path = "road_1080p.mp4"

original_folder = "original_frames"
denoised_folder ="denoised_frames"
contrast_folder = "contrast_clahe_frames"

os.makedirs(original_folder, exist_ok=True)
os.makedirs(denoised_folder, exist_ok=True)
os.makedirs(contrast_folder, exist_ok=True)

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

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

count = 0

for filename in sorted(os.listdir(denoised_folder)):

    if filename.endswith(".jpg") or filename.endswith(".png"):

        img_path = os.path.join(denoised_folder, filename)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        p2, p98 = np.percentile(img, (2, 98))
        contrast = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        enhanced = clahe.apply(contrast)

        output_path = os.path.join(contrast_folder, f"enhanced_{filename}")
        cv2.imwrite(output_path, enhanced)

        count += 1

print(f"Done! {count} enhanced frames saved in '{contrast_folder}'")
