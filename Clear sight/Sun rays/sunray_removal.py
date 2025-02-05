import cv2
import numpy as np

def remove_sun_rays(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Convert to grayscale to focus on intensity
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to detect bright regions (possible sunrays)
    thresholded = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    # Refine the mask using morphological operations (remove small noises)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    refined_mask = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    # Further refine the mask to isolate sunray-like structures
    refined_mask = cv2.dilate(refined_mask, kernel, iterations=1)

    # Inpainting to remove sun rays (only in the identified regions)
    inpainted_image = cv2.inpaint(image, refined_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    # Blend the inpainted image with the original image (keep the unaffected areas intact)
    final_image = cv2.bitwise_and(inpainted_image, inpainted_image, mask=cv2.bitwise_not(refined_mask))
    final_image += cv2.bitwise_and(image, image, mask=refined_mask)

    # Save the result
    cv2.imwrite(output_path, final_image)
    print(f"Sunray removed image saved at: {output_path}")

# Example usage
input_image = r"C:\Users\niros\OneDrive\Desktop\EE 405 FYP clear sight and TSR\Clear sight\Sun rays\sun_ray.jpg"
output_image = r"C:\Users\niros\OneDrive\Desktop\EE 405 FYP clear sight and TSR\Clear sight\Sun rays\sunray_removed.jpg"
remove_sun_rays(input_image, output_image)
