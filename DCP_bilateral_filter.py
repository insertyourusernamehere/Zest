import os
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing

def dark_channel_prior(img, window_size=15):    
    min_channel = np.min(img, axis=2)
    dark_channel = cv2.erode(min_channel, np.ones((window_size, window_size), dtype=np.uint8))
    return dark_channel

def dehaze_image(img_path, output_folder):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # Convert to float32 in the range [0, 1]

    window_size = 20
    omega = 0.85
    t0 = 0.05

    A = np.percentile(img, 99, axis=(0,1))

    # Convert image to float32 in range [0, 1] for bilateral filtering
    img_float32 = img.astype(np.float32)

    # Apply bilateral filter to the image
    bilateral_filtered_img = cv2.bilateralFilter(img_float32, 9, 75, 75)

    # Compute the dark channel prior on the filtered image
    transmission = 1 - omega * dark_channel_prior(bilateral_filtered_img, window_size)
    
    # Resize transmission to match the dimensions of img
    transmission_resized = cv2.resize(transmission, (img.shape[1], img.shape[0]))

    dehazed_img = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        dehazed_img[:,:,i] = (img[:,:,i] - A[i]) / np.maximum(transmission_resized, t0) + A[i]
    dehazed_img = np.clip(dehazed_img, 0, 1.0)

    # Convert back to uint8 for saving
    dehazed_img_uint8 = (dehazed_img * 255.0).astype(np.uint8)

    filename = os.path.basename(img_path)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, cv2.cvtColor(dehazed_img_uint8, cv2.COLOR_RGB2BGR))

def dehaze_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_images = [f for f in os.listdir(input_folder) if f.endswith(".jpg") or f.endswith(".png")]
    for filename in tqdm(input_images, desc="Dehazing", unit="image"):
        img_path = os.path.join(input_folder, filename)
        dehaze_image(img_path, output_folder)

if __name__ == "__main__":
    cv2.setNumThreads(multiprocessing.cpu_count())
    input_folder = r'C:\Users\HP\Desktop\IIST\inputs'
    output_folder = r'C:\Users\HP\Desktop\IIST\bilateral_filtered'
    dehaze_images(input_folder, output_folder)
    print("DCP complete.")
