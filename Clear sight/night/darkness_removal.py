import cv2
import numpy as np

def enhance_night_image(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Convert to LAB color space for brightness control
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to enhance local contrast in the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Merge channels back and convert to BGR
    lab_clahe = cv2.merge((l_clahe, a, b))
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Convert image to float32 for log transformation
    image_float = np.float32(image_clahe) / 255.0

    # Apply logarithmic compression to reduce highlights
    log_transformed = np.log1p(image_float) / np.log(1.0 + 1.0)  # Normalize with log(1 + max_value)
    log_transformed = np.clip(log_transformed, 0, 1)

    # Scale back to 8-bit
    compressed_image = (log_transformed * 255).astype("uint8")

    # Apply bilateral filtering to reduce noise and preserve edges
    enhanced_image = cv2.bilateralFilter(compressed_image, d=9, sigmaColor=75, sigmaSpace=75)

    # Save the output
    cv2.imwrite(output_path, enhanced_image)
    print(f"Enhanced night image saved at: {output_path}")

# Example usage
input_image = "C:\\Users\\niros\\OneDrive\\Desktop\\EE 405 FYP clear sight and TSR\\Clear sight\\night\\night_image.jpg"
output_image = "C:\\Users\\niros\\OneDrive\\Desktop\\EE 405 FYP clear sight and TSR\\Clear sight\\night\\night_image_enhanced.jpg"
enhance_night_image(input_image, output_image)
