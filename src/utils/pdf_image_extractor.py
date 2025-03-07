# src/utils/pdf_image_extractor.py

import fitz
from pathlib import Path
import shutil
import logging
import csv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parameters
DPI = 200  # Adjust as needed
INFILE = "manuals_to_sets.csv"


def extract_pages_as_images(
    pdf_path: Path,
    images_dir: Path,
    set_num: str,
    booklet_num: str,
    dpi: int = 300,
) -> list[Path]:
    """
    Extract each page of a PDF as a PNG image.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory where images will be saved
        dpi: Resolution for the extracted images (default: 300)

    Returns:
        List of paths to extracted image files
    """
    # Create output directory if it doesn't exist
    images_dir.mkdir(parents=True, exist_ok=True)

    # Calculate zoom factor based on DPI (default PDF DPI is 72)
    zoom = dpi / 72

    created_files = []

    # Open PDF
    pdf_document = fitz.open(pdf_path)

    try:
        num_pages = len(pdf_document)
        padding = len(str(num_pages))
        # Iterate through pages
        for page_num, page in enumerate(pdf_document):
            # Create matrix for better resolution
            # mat = fitz.Matrix(zoom, zoom)

            # Get page pixmap (image)
            # pix = page.get_pixmap(matrix=mat)
            pix = page.get_pixmap()

            # Generate output path
            page_num_str = str(page_num + 1).zfill(padding)
            image_path = (
                images_dir / f"{set_num}_{booklet_num}_{page_num_str}.jpg"
            )

            # Save image
            pix.save(image_path)
            created_files.append(image_path)
            logger.info(f"Saved page {page_num + 1} as: {image_path}")

    finally:
        pdf_document.close()

    logger.info(f"\nExtracted {len(created_files)} pages as images")
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
    extracted_images_dir = project_root / "data" / "images"

    # Create the directory if it doesn't exist
    extracted_images_dir.mkdir(parents=True, exist_ok=True)

    # Get the list of PDF files
    with open(
        project_root / "data" / "training" / "manuals" / INFILE, mode="r"
    ) as infile:
        reader = csv.reader(infile)
        for row in reader:
            if not row:
                continue
            pdf_path = (
                project_root
                / "data"
                / "training"
                / "manuals"
                / f"{row[0]}.pdf"
            )
            set_num = row[1]
            booklet_num = row[2]
            # Extract images
            if pdf_path.exists():
                created_files = extract_pages_as_images(
                    pdf_path=pdf_path,
                    images_dir=extracted_images_dir,
                    set_num=set_num,
                    booklet_num=booklet_num,
                    dpi=DPI,
                )

                # Print summary of created files
                logger.info("\nCreated/Modified files during execution:")
                for file_path in created_files:
                    logger.info(file_path)
            else:
                logger.error(f"PDF file not found: {pdf_path}")


if __name__ == "__main__":
    main()
