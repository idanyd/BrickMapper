import pytesseract
from PIL import ImageEnhance
import matplotlib.pyplot as plt

import logging

logger = logging.getLogger(__name__)


def preprocess_image(image):
    # Convert to grayscale
    img = image.convert("L")

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    # Enhance brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.5)

    return img


def display_images(orig_image, processed_image):
    # Display original and processed images side by side

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(orig_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Processed Image")
    plt.imshow(processed_image, cmap="gray")
    plt.axis("off")

    plt.show()


def extract_numbers(image, display_images=False):

    # Preprocess the image
    processed_image = preprocess_image(image)

    if display_images:
        display_images(image, processed_image)

    # Configure tesseract to only look for digits (0-9)
    custom_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"

    # Extract text from image
    try:
        number = pytesseract.image_to_string(
            processed_image, config=custom_config
        ).strip()
        return int(number)

    except Exception as e:
        logger.warning(f"Couldn't extract number from step box: {str(e)}")
        return None
