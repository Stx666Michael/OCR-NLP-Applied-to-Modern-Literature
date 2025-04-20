import os
import cv2
import numpy as np


def apply_augmentation(image):
    # Randomly rotate the image by a random angle
    angle = np.random.randint(-10, 10)
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (cols, rows), borderValue=255)

    # Randomly shift the image horizontally and vertically
    max_shift = int(min(rows, cols) * 0.1)
    dx = np.random.randint(-max_shift, max_shift)
    dy = np.random.randint(-max_shift, max_shift)
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    image = cv2.warpAffine(image, translation_matrix, (cols, rows), borderValue=255)

    # Apply salt and pepper noise
    noise_probability = 0.05
    salt_pepper_mask = np.random.rand(*image.shape)
    image[salt_pepper_mask < noise_probability / 2] = 255
    image[salt_pepper_mask > 1 - noise_probability / 2] = 0

    return image


def extract_patch(image, x, y, x_shift, y_shift, output_path, augmentation, res_ratio):
    # Extract the patch
    patch = image[int(y):int(y+y_shift), int(x):int(x+x_shift)]

    if augmentation:
        patch = apply_augmentation(patch)

    if res_ratio != 1:
        patch = cv2.resize(patch, (int(x_shift*res_ratio), int(y_shift*res_ratio)))
    
    # Write the patch as a new image
    cv2.imwrite(output_path, patch)


def build_dataset(input_path, char_per_row, char_per_col, num_of_chars, output_path, font_index, augmentation=False, res_ratio=1):
    # Read the image
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    height, width = image.shape
    x_shift = width / char_per_row
    y_shift = height / char_per_col
    char_index = 0

    for i in range(char_per_col):
        for j in range(char_per_row):
            extract_patch(image, j*x_shift, i*y_shift, x_shift, y_shift, f"{output_path}{char_index}_{font_index}.jpg", augmentation, res_ratio)
            char_index += 1
            if char_index >= num_of_chars:
                return
            

if __name__ == "__main__":
    augmentation = False
    res_ratio = 1
    output_path = "data/train_6k/"
    char_per_row = 74
    char_per_col = 82
    num_of_chars = 6000
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in range(12):
        input_path = f"data/source/CCL6k-{i+1}.png"
        font_index = i
        build_dataset(input_path, char_per_row, char_per_col, num_of_chars, output_path, font_index, augmentation, res_ratio)
        print(f"Font {i+1} done")
    print("Dataset built in", output_path)