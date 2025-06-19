import cv2
import numpy as np
import matplotlib.pyplot as plt

def reduce_intensity_levels(image, levels):
    """Reduce intensity levels to a specified number (power of 2)."""
    if not (2 <= levels <= 256 and (levels & (levels - 1) == 0)):
        raise ValueError("Levels must be a power of 2 between 2 and 256.")
    factor = 256 // levels
    reduced_image = (image // factor) * factor
    return reduced_image

# Load image in grayscale
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found.")

# Example: reduce to 4 levels
levels = 4  # User input (must be power of 2)
result = reduce_intensity_levels(image, levels)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(1, 2, 2), plt.imshow(result, cmap='gray'), plt.title(f'{levels} Intensity Levels')
plt.show()

# Save result
cv2.imwrite(f'reduced_levels_{levels}.png', result)