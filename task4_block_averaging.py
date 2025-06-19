import cv2
import numpy as np
import matplotlib.pyplot as plt

def reduce_block_resolution(image, block_size):
    """Replace each non-overlapping block with its average."""
    h, w = image.shape
    h_new, w_new = h // block_size, w // block_size
    result = np.zeros((h_new * block_size, w_new * block_size), dtype=image.dtype)

    for i in range(0, h_new):
        for j in range(0, w_new):
            block = image[i * block_size:(i+1) * block_size, j * block_size:(j+1) * block_size]
            avg = np.mean(block).astype(image.dtype)
            result[i * block_size:(i+1) * block_size, j * block_size:(j+1) * block_size] = avg
    return result

# Load image in grayscale
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found.")

# Apply block averaging
result_3x3 = reduce_block_resolution(image, 3)
result_5x5 = reduce_block_resolution(image, 5)
result_7x7 = reduce_block_resolution(image, 7)

# Display results
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(1, 4, 2), plt.imshow(result_3x3, cmap='gray'), plt.title('3x3 Blocks')
plt.subplot(1, 4, 3), plt.imshow(result_5x5, cmap='gray'), plt.title('5x5 Blocks')
plt.subplot(1, 4, 4), plt.imshow(result_7x7, cmap='gray'), plt.title('7x7 Blocks')
plt.show()

# Save results
cv2.imwrite('blocks_3x3.png', result_3x3)
cv2.imwrite('blocks_5x5.png', result_5x5)
cv2.imwrite('blocks_7x7.png', result_7x7)