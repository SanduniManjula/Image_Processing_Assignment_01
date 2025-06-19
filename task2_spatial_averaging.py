import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_average_filter(image, kernel_size):
    """Apply average filter with specified kernel size."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

# Load image in grayscale
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found.")

# Apply filters for different kernel sizes
result_3 = apply_average_filter(image, 3)
result_10 = apply_average_filter(image, 10)
result_20 = apply_average_filter(image, 20)

# Display results
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(1, 4, 2), plt.imshow(result_3, cmap='gray'), plt.title('3x3 Filter')
plt.subplot(1, 4, 3), plt.imshow(result_10, cmap='gray'), plt.title('10x10 Filter')
plt.subplot(1, 4, 4), plt.imshow(result_20, cmap='gray'), plt.title('20x20 Filter')
plt.show()

# Save results
cv2.imwrite('filter_3x3.png', result_3)
cv2.imwrite('filter_10x10.png', result_10)
cv2.imwrite('filter_20x20.png', result_20)