import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def gamma_correction(img, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def image_metrics(img):
    mean_val = np.mean(img)
    std_val = np.std(img)
    return mean_val, std_val

video_path = "road_1080p.mp4"

original_folder = "original_frames"
gray_folder = "gray_frames"
gaussian_folder ="gaussian_frames"
median_folder="median_folder"
contrast_folder = "contrast_frames"
clahe_folder="clahe_frames"
gamma_folder ="gamma_frames"
lap_folder="lap_frames"

os.makedirs(original_folder, exist_ok=True)
os.makedirs(gray_folder, exist_ok=True)
os.makedirs(gaussian_folder, exist_ok=True)
os.makedirs(median_folder, exist_ok=True)
os.makedirs(contrast_folder, exist_ok=True)
os.makedirs(clahe_folder, exist_ok=True)
os.makedirs(gamma_folder, exist_ok=True)
os.makedirs(lap_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 1
f = 0

results = []

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
        clahe_enhanced = clahe.apply(contrast)

        gamma_img = gamma_correction(clahe_enhanced, 1.2)

        lap = cv2.Laplacian(gamma_img, cv2.CV_64F)
        lap = cv2.convertScaleAbs(lap)
        lap = gamma_img - lap

        cv2.imwrite(f"{gray_folder}/gray_frame_{frame_count:05d}.jpg", gray)
        cv2.imwrite(f"{gaussian_folder}/gaussian_frame_{frame_count:05d}.jpg", gaussian)
        cv2.imwrite(f"{median_folder}/median_frame_{frame_count:05d}.jpg", median)
        cv2.imwrite(f"{contrast_folder}/contrast_frame_{frame_count:05d}.jpg", contrast)
        cv2.imwrite(f"{clahe_folder}/clahe_frame_{frame_count:05d}.jpg", clahe_enhanced)
        cv2.imwrite(f"{gamma_folder}/gamma_frame_{frame_count:05d}.jpg", gamma_img)
        cv2.imwrite(f"{lap_folder}/lap_frame_{frame_count:05d}.jpg", lap)

        mean_before, std_before = image_metrics(gray)
        mean_after, std_after = image_metrics(lap)

        results.append([frame_count, mean_before, mean_after, std_before, std_after])

        f += 1

    frame_count += 1

cap.release()

print("\nEnhancement Result Table")
print("Frame\tMean Before\tMean After\tStd Before\tStd After")

for r in results:
    print(f"{r[0]}\t{r[1]:.2f}\t\t{r[2]:.2f}\t\t{r[3]:.2f}\t\t{r[4]:.2f}")

folders = [
    ("Original", original_folder),
    ("Gray", gray_folder),
    ("Gaussian", gaussian_folder),
    ("Median", median_folder),
    ("Contrast", contrast_folder),
    ("CLAHE", clahe_folder),
    ("Gamma", gamma_folder),
    ("Laplacian", lap_folder)
]

images = []

for name, folder in folders:
    files = sorted(os.listdir(folder))
    first_img_path = os.path.join(folder, files[0])
    img = cv2.imread(first_img_path, 0)
    images.append((name, img))

plt.figure(figsize=(16,20))

for i, (name, img) in enumerate(images):

    plt.subplot(8,2,2*i+1)
    plt.imshow(img, cmap='gray')
    plt.title(name)
    plt.axis("off")

    plt.subplot(8,2,2*i+2)
    plt.hist(img.ravel(),256,[0,256])
    plt.title(name + " Histogram")

plt.tight_layout()
plt.show()

input_img = cv2.imread("gray_frames/gray_frame_00041.jpg")
output_img = cv2.imread("lap_frames/lap_frame_00041.jpg")

cv2.namedWindow("input", cv2.WINDOW_NORMAL)
cv2.resizeWindow("input", 800, 500)

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("output", 800, 500)

cv2.imshow("input", input_img)
cv2.imshow("output", output_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Done! {f} frames saved.")

