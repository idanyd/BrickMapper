# src/utils/pdf_image_extractor.py

import fitz
from pathlib import Path
import shutil
import logging
import csv
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_pages_as_images(
    pdf_path: Path,
    images_dir: Path,
    set_num: str,
    booklet_num: str,
) -> list[Path]:
    """
    Extract each page of a PDF as a PNG image.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory where images will be saved

    Returns:
        List of paths to extracted image files
    """
    # Create output directory if it doesn't exist
    images_dir.mkdir(parents=True, exist_ok=True)

    created_files = []

    # Open PDF
    pdf_document = fitz.open(pdf_path)

    try:
        num_pages = len(pdf_document)
        padding = len(str(num_pages))
        # Iterate through pages
        for page_num, page in enumerate(pdf_document):

            # Get page pixmap (image)
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

    logger.info(f"Extracted {len(created_files)} pages as images")
    return created_files


def extract_pages_as_images_generator(pdf_path: Path):
    """
    Extract each page of a PDF as a PNG image and yield them one by one.

    Args:
        pdf_path: Path to the PDF file
        set_num: Set number for filename
        booklet_num: Booklet number for filename

    Yields:
        Tuple of (page_number, image_data) for each page
    """
    # Open PDF
    pdf_document = fitz.open(pdf_path)

    try:
        # Iterate through pages
        for page_num, page in enumerate(pdf_document):
            # Get page pixmap (image)
            pix = page.get_pixmap()

            # Convert to PIL Image
            img_bytes = pix.tobytes("jpeg")
            pil_image = Image.open(io.BytesIO(img_bytes))

            # Yield the image data and metadata
            yield {
                "page_num": page_num + 1,
                "image_data": pil_image,
            }

    finally:
        pdf_document.close()


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


def extract_pieces_from_step(pdf_path, page_num, step_info):
    doc = fitz.open(pdf_path)
    images = []

    page = doc[page_num]

    # Get all images on the page
    for img in page.get_images(full=True):
        xref = img[0]
        base_image = doc.extract_image(xref)

        if base_image["width"] < 10 or base_image["height"] < 10:
            # Skip small images
            continue

        # Get the image rectangle (bbox) from the page
        bbox = None
        for rect in page.get_image_rects(xref):
            # Just take the first occurrence's bbox
            bbox = rect
            break

        # Only add images that are in the step bounding box
        if (
            bbox
            and bbox[0] > step_info[0]
            and bbox[1] > step_info[1]
            and bbox[2] < step_info[2]
            and bbox[3] < step_info[3]
        ):

            images.append(base_image["image"])

    doc.close()
    return images


def main():
    # Parameters
    INFILE = "manuals_to_sets.csv"

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
