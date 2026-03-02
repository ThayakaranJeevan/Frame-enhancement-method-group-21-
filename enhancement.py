import cv2
import os
import numpy as np

def gamma_correction(img, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

video_path = "road_1080p.mp4"

original_folder = "original_frames"
denoised_folder ="denoised_frames"
contrast_folder = "contrast_clahe_frames"
sharpened_folder ="sharpened_folder_frames"

os.makedirs(original_folder, exist_ok=True)
os.makedirs(denoised_folder, exist_ok=True)
os.makedirs(contrast_folder, exist_ok=True)
os.makedirs(sharpened_folder, exist_ok=True)

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

        p2, p98 = np.percentile(median, (2, 98))
        contrast = np.clip(median, p2, p98)
        contrast = ((contrast - p2) / (p98 - p2) * 255).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(contrast)

        gamma_img = gamma_correction(enhanced, 1.2)
        sharpened = cv2.addWeighted(gamma_img, 1.5, gamma_img, -0.5, 0)

        frame_name = f"{denoised_folder}/denoised_frame_{frame_count:05d}.jpg"
        cv2.imwrite(frame_name, median)

        frame_name = f"{contrast_folder}/contrast_clahe_frame_{frame_count:05d}.jpg"
        cv2.imwrite(frame_name, enhanced)

        frame_name = f"{sharpened_folder}/sharpened_frame_{frame_count:05d}.jpg"
        cv2.imwrite(frame_name, sharpened)
        
        f += 1

    frame_count += 1

cap.release()

print(f"Done! {f} frames saved.")

