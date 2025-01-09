# src/utils/pdf_image_extractor.py

import fitz
from pathlib import Path
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_data_directories(base_dir: Path) -> tuple[Path, Path]:
    """
    Set up the necessary data directories.

    Args:
        base_dir: Base project directory path

    Returns:
        Tuple of (data_dir, extracted_images_dir)
    """
    # Create main data directory structure
    data_dir = base_dir / "data"
    extracted_images_dir = data_dir / "extracted_images"

    # Create directories if they don't exist
    for dir_path in [data_dir, extracted_images_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified directory: {dir_path}")

    return data_dir, extracted_images_dir

def extract_images_from_pdf(
    pdf_path: Path,
    output_dir: Path,
    manual_name: str = None
) -> list[Path]:
    """
    Extract images from a PDF file.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory where images will be saved
        manual_name: Optional name of the LEGO manual (for organizing images)

    Returns:
        List of paths to extracted image files
    """
    # Create manual-specific subdirectory if name provided
    if manual_name:
        output_dir = output_dir / manual_name
        output_dir.mkdir(parents=True, exist_ok=True)

    # Open the PDF
    pdf_document = fitz.open(pdf_path)
    created_files = []

    try:
        # Counter for images
        image_count = 0

        # Iterate through each page
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]

            # Get images from page
            images = page.get_images()

            # Iterate through images
            for image_index, img in enumerate(images, start=1):
                # Get image data
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]

                # Get image extension
                ext = base_image["ext"]

                # Generate image file name
                if manual_name:
                    image_filename = f"{manual_name}_page_{page_num + 1}_img_{image_index}.{ext}"
                else:
                    image_filename = f"page_{page_num + 1}_img_{image_index}.{ext}"

                image_path = output_dir / image_filename

                # Save image
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)
                    image_count += 1
                    created_files.append(image_path)
                    logger.info(f"Saved: {image_filename}")

        logger.info(f"\nExtracted {image_count} images from the PDF")

    finally:
        pdf_document.close()

    return created_files

def clean_output_directory(output_dir: Path):
    """
    Clean the output directory by removing all existing files.

    Args:
        output_dir: Directory to clean
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info(f"Cleaned directory: {output_dir}")
    output_dir.mkdir(parents=True)

def main():
    # Get project root directory (assuming script is in src/utils)
    project_root = Path(__file__).parent.parent.parent

    # Setup directories
    extracted_images_dir = setup_data_directories(project_root)

    # Example usage
    pdf_path = project_root / "manuals" / "example.pdf"  # Replace with your PDF path
    manual_name = "example_manual"  # Replace with your manual name

    # Clean output directory (optional)
    # clean_output_directory(extracted_images_dir)

    # Extract images
    if pdf_path.exists():
        created_files = extract_images_from_pdf(
            pdf_path=pdf_path,
            output_dir=extracted_images_dir,
            manual_name=manual_name
        )

        # Print summary of created files
        logger.info("\nCreated/Modified files during execution:")
        for file_path in created_files:
            logger.info(file_path)
    else:
        logger.error(f"PDF file not found: {pdf_path}")

if __name__ == "__main__":
    main()