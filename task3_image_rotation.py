import cv2
import numpy as np
import matplotlib.pyplot as plt

def rotate_image(image, angle):
    """Rotate image by specified angle in degrees."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image

# Load image in grayscale
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found.")

# Rotate by 45° and 90°
result_45 = rotate_image(image, 45)
result_90 = rotate_image(image, 90)

# Display results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(1, 3, 2), plt.imshow(result_45, cmap='gray'), plt.title('Rotated 45°')
plt.subplot(1, 3, 3), plt.imshow(result_90, cmap='gray'), plt.title('Rotated 90°')
plt.show()

# Save results
cv2.imwrite('rotated_45.png', result_45)
cv2.imwrite('rotated_90.png', result_90)