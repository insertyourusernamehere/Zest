import os
import cv2
import numpy as np
import argparse
import cv2.dnn
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml
from tqdm import tqdm

CLASSES = yaml_load(check_yaml("data.yaml"))["names"]

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
    return output_path

def dehaze_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_images = [f for f in os.listdir(input_folder) if f.endswith(".jpg") or f.endswith(".png")]
    progress_bar = tqdm(total=len(input_images), desc="Processing images", unit="image")
    for filename in input_images:
        img_path = os.path.join(input_folder, filename)
        output_path = dehaze_image(img_path, output_folder)
        progress_bar.update(1)
    progress_bar.close()

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

def process_images_with_gamma_correction(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_images = [f for f in os.listdir(input_folder) if f.endswith(".jpg") or f.endswith(".png")]
    progress_bar = tqdm(total=len(input_images), desc="Processing images", unit="image")
    for filename in input_images:
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Unable to read {filename}")
            continue

        darkness_amount = calculate_darkness(image)
        corrected_image = gamma_correction(image, darkness_amount)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, corrected_image)
        progress_bar.update(1)
    progress_bar.close()

def write_pixel_values(file_path, detections, scale):
    with open(file_path, "w") as file:
        for detection in detections:
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            box = detection["box"]
            x1, y1, w, h = box
            x1 = int(x1 * scale)
            y1 = int(y1 * scale)
            w = int(w * scale)
            h = int(h * scale)
            x2 = x1 + w
            y2 = y1 + h
            line = f"{class_name}: {x1} {y1} {x2} {y2}\n"
            file.write(line)

def detect_objects(onnx_model, input_image):
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)
    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape

    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    scale = length / 640
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)
    outputs = model.forward()
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.45:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []

    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)

    return detections

def generate_text_file(image_path, output_folder):
    filename = os.path.basename(image_path)
    output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
    detections = detect_objects(onnx_model, image_path)

    if detections:
        write_pixel_values(output_file_path, detections, detections[0]["scale"])

def main(onnx_model, input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_images = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    progress_bar = tqdm(total=len(input_images), desc="Total Progress", unit="image")

    for image_file in input_images:
        input_image_path = os.path.join(input_folder, image_file)

        # Dehaze the image
        dehazed_image_path = dehaze_image(input_image_path, output_folder)

        # Apply gamma correction
        gamma_corrected_path = os.path.join(output_folder, image_file)
        image = cv2.imread(dehazed_image_path)
        darkness_amount = calculate_darkness(image)
        corrected_image = gamma_correction(image, darkness_amount)
        cv2.imwrite(gamma_corrected_path, corrected_image)

        # Generate text file with object detections
        generate_text_file(gamma_corrected_path, output_folder)

        # Remove the dehazed image
        os.remove(dehazed_image_path)

        progress_bar.update(1)

    progress_bar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best.onnx", help="Input your ONNX model.")
    parser.add_argument("--input-folder", required=True, help="Path to the input folder containing images.")
    parser.add_argument("--output-folder", default="results", help="Path to the output folder to save detection results.")
    args = parser.parse_args()
    onnx_model = args.model
    input_folder = args.input_folder
    output_folder = args.output_folder

    main(onnx_model, input_folder, output_folder)