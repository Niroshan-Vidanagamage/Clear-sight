import cv2
import numpy as np

def remove_rain_advanced(image_path, output_path):
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise and highlight streaks
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using the Sobel operator
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Normalize and threshold the gradient to isolate rain streaks
    _, binary_streaks = cv2.threshold((sobel_magnitude / sobel_magnitude.max() * 255).astype(np.uint8), 30, 255, cv2.THRESH_BINARY)

    # Dilate the streaks slightly for better inpainting
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_streaks = cv2.dilate(binary_streaks, kernel, iterations=1)

    # Inpaint the image to remove detected rain streaks
    inpainted = cv2.inpaint(image, dilated_streaks, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    # Apply bilateral filtering to smooth remaining noise while preserving edges
    smoothed = cv2.bilateralFilter(inpainted, d=9, sigmaColor=75, sigmaSpace=75)

    # Enhance contrast using CLAHE (adaptive histogram equalization)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_image = cv2.merge((l, a, b))
    final_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)

    # Save the result
    cv2.imwrite(output_path, final_image)
    print(f"Rain-removed image saved at: {output_path}")

# Example usage
input_image = "C:\\Users\\niros\\OneDrive\\Desktop\\EE 405 FYP clear sight and TSR\\Clear sight\\rain\\rainy_image1.jpg"  # Replace with your input image path
output_image = "C:\\Users\\niros\\OneDrive\\Desktop\\EE 405 FYP clear sight and TSR\\Clear sight\\rain\\rain_removed_image3.jpg"  # Replace with your desired output path
remove_rain_advanced(input_image, output_image)
