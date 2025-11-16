import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=== Image Quality Enhancement Test ===")
print("All imports successful!")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")

# Create a simple test image
test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
plt.imshow(test_image)
plt.title("Test Image - Libraries Working!")
plt.axis('off')
plt.show()

print("Test completed successfully!")
