import cv2
import numpy as np
import os
from tqdm import tqdm
import multiprocessing

def calculate_darkness(image):
    # Compute the average pixel intensity
    average_intensity = image.mean()

    # Map the average intensity to the desired range (0.5 to 3)
    darkness_amount = (average_intensity / 255.0) * 2.5 + 0.5

    # Ensure darkness amount is within the specified range
    darkness_amount = min(max(darkness_amount, 0.5), 3.0)

    return darkness_amount - 0.2


def gamma_correction(image, darkness_amount):

    # Apply gamma correction formula
    gamma_inv =  darkness_amount
    table = (np.arange(0, 256) / 255.0) ** gamma_inv * 255
    return cv2.LUT(image, table.astype(np.uint8))

def main():
    cv2.setNumThreads(multiprocessing.cpu_count())
    input_folder = r'C:\Users\HP\Desktop\IIST\bilateral_filtered'  # Input folder containing images
    output_folder = r'C:\Users\HP\Desktop\IIST\bilateral_filtered_gamma_corrected'  # Output folder for corrected images


    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder with tqdm progress bar
    for filename in tqdm(os.listdir(input_folder), desc="Processing images"):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Filter only JPEG and PNG files
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error: Unable to read {filename}")
                continue

            # Calculate darkness amount
            darkness_amount = calculate_darkness(image)

            # Apply gamma correction based on darkness amount
            corrected_image = gamma_correction(image, darkness_amount)

            # Save the corrected image to the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, corrected_image)      

    print("Gamma correction complete.")  

if __name__ == "__main__":
    main()
