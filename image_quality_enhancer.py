import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_preprocess_image(image_path):
    """Load and convert image to RGB format"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def apply_histogram_equalization(image):
    """Apply Histogram Equalization for contrast improvement"""
    image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
    equalized_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    return equalized_image

def apply_clahe(image, clip_limit=2.0, grid_size=(8,8)):
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)"""
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    image_lab[:,:,0] = clahe.apply(image_lab[:,:,0])
    clahe_image = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)
    return clahe_image

def apply_sharpening(image, strength=1.0):
    """Apply image sharpening using kernel convolution"""
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]]) * strength
    sharpened_image = cv2.filter2D(image, -1, kernel)
    sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
    return sharpened_image

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Apply bilateral filter for noise reduction while preserving edges"""
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return filtered_image

def create_sample_image():
    """Create a sample test image with various features"""
    height, width = 400, 600
    sample_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient background
    for i in range(height):
        for j in range(width):
            sample_image[i, j] = [j//3, i//2, (i+j)//3]
    
    # Add some shapes for testing
    cv2.rectangle(sample_image, (50, 50), (200, 150), (255, 0, 0), -1)
    cv2.circle(sample_image, (400, 100), 60, (0, 255, 0), -1)
    cv2.ellipse(sample_image, (300, 250), (100, 50), 45, 0, 360, (0, 0, 255), -1)
    
    # Add some text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(sample_image, 'TEST IMAGE', (150, 350), font, 1, (255, 255, 255), 2)
    cv2.putText(sample_image, 'OpenCV Demo', (180, 380), font, 0.7, (200, 200, 200), 2)
    
    return sample_image

def main():
    print("🚀 Image Quality Enhancement Tool")
    print("=" * 50)
    
    # Look for image files in current directory
    image_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if image_files:
        image_path = image_files[0]
        print(f" Found image: {image_path}")
        try:
            original_image = load_and_preprocess_image(image_path)
            print(f" Image loaded successfully! Dimensions: {original_image.shape}")
        except Exception as e:
            print(f" Error loading {image_path}: {e}")
            print(" Creating a sample image instead...")
            original_image = create_sample_image()
            image_path = "sample_image.jpg"
            cv2.imwrite(image_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
            print(f"📸 Created sample image: {image_path}")
    else:
        print(" No image files found. Creating a sample image...")
        original_image = create_sample_image()
        image_path = "sample_image.jpg"
        cv2.imwrite(image_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        print(f" Created sample image: {image_path}")
    
    try:
        print("\n Applying enhancement techniques...")
        
        # Apply enhancement techniques
        equalized_img = apply_histogram_equalization(original_image)
        clahe_img = apply_clahe(original_image)
        sharpened_img = apply_sharpening(original_image)
        bilateral_img = apply_bilateral_filter(original_image)
        
        print("py All enhancements applied successfully!")
        
        # Display results
        images = [original_image, equalized_img, clahe_img, sharpened_img, bilateral_img]
        titles = [
            'Original Image', 
            'Histogram Equalization\n(Global Contrast)', 
            'CLAHE\n(Local Contrast)', 
            'Sharpening\n(Edge Enhancement)', 
            'Bilateral Filter\n(Noise Reduction)'
        ]
        
        # Create a beautiful comparison plot
        plt.figure(figsize=(18, 10))
        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(2, 3, i+1)
            plt.imshow(img)
            plt.title(title, fontsize=12, fontweight='bold', pad=10)
            plt.axis('off')
        
        plt.suptitle('Image Quality Enhancement Techniques Comparison', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 ENHANCEMENT TECHNIQUES APPLIED:")
        print("=" * 60)
        print("1. 📈 HISTOGRAM EQUALIZATION")
        print("   - Improves global contrast")
        print("   - Spreads out intensity values")
        
        print("\n2. CLAHE (Contrast Limited Adaptive Histogram Equalization)")
        print("   - Local contrast enhancement") 
        print("   - Prevents noise amplification")
        
        print("\n3.  SHARPENING")
        print("   - Enhances edge definition")
        print("   - Makes details more visible")
        
        print("\n4.  BILATERAL FILTER")
        print("   - Reduces noise while preserving edges")
        print("   - Non-linear, edge-preserving smoothing")
        
        print(f"\n Original image: {image_path}")
        print(f" Image size: {original_image.shape[1]} x {original_image.shape[0]} pixels")
        print(" Enhancement completed successfully!")
        
    except Exception as e:
        print(f" Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()