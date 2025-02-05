import cv2
import numpy as np

def remove_fog(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Normalize the image to the range [0, 1]
    normalized_img = image.astype(np.float32) / 255.0

    # Calculate the dark channel
    kernel_size = 15  # Size of the patch to compute the dark channel
    min_channel = np.min(normalized_img, axis=2)
    dark_channel = cv2.erode(min_channel, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)))

    # Estimate the atmospheric light
    flat_dark_channel = dark_channel.ravel()
    flat_image = normalized_img.reshape(-1, 3)
    num_pixels = len(flat_dark_channel)
    num_brightest = max(num_pixels // 1000, 1)  # Top 0.1% brightest pixels
    indices = np.argsort(flat_dark_channel)[-num_brightest:]
    atmospheric_light = np.mean(flat_image[indices], axis=0)

    # Calculate the transmission map
    omega = 0.95  # Scattering coefficient
    transmission = 1 - omega * dark_channel / atmospheric_light.max()

    # Refine the transmission map using median blur
    transmission_refined = cv2.medianBlur((transmission * 255).astype(np.uint8), 15) / 255.0

    # Restore the image
    t = np.clip(transmission_refined, 0.1, 1)  # Avoid division by zero
    dehazed_img = (normalized_img - atmospheric_light) / t[:, :, None] + atmospheric_light
    dehazed_img = np.clip(dehazed_img, 0, 1)

    # Convert back to 8-bit
    output_image = (dehazed_img * 255).astype(np.uint8)

    # Save the dehazed image
    cv2.imwrite(output_path, output_image)
    print(f"Dehazed image saved at: {output_path}")

# Example usage
input_image = r"C:\Users\niros\OneDrive\Desktop\EE 405 FYP clear sight and TSR\Clear sight\fog\foggy_image.webp"
output_image = r"C:\Users\niros\OneDrive\Desktop\EE 405 FYP clear sight and TSR\Clear sight\fog\fog_removed_image.jpg"
remove_fog(input_image, output_image)
