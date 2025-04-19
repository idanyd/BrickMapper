from pathlib import Path
from PIL import Image
import numpy as np
import io

import logging

logger = logging.getLogger(__name__)


def save_images(output_dir, images, filename_format="{xref}.{ext}"):
    """
    Save multiple images with dynamically formatted filenames.

    Args:
        output_dir: Directory to save the images in (default: current directory)
        images: List of dictionaries, each containing data for one image
        filename_format: A string format for the filenames (uses keys from each image dict)

    Returns:
        List of saved filenames
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_filenames = []

    for img in images:
        try:
            filename = filename_format.format(**img)
            img = Image.fromarray(img["image_data"])
            img.save(output_dir / filename)

            saved_filenames.append(filename)
        except KeyError as e:
            logger.error(f"Key error in filename formatting: {e}")

    return saved_filenames


def is_complex_image(img_array):
    """
    Determine if an image is complex enough to be a LEGO piece render
    rather than a simple bar or solid color.
    """
    # Get the first pixel value
    first_pixel = img_array[0, 0]

    # Compare all pixels to the first one
    # If any pixel is different from the first one, it's complex
    return np.any(img_array != first_pixel)


def extract_element_ids_and_positions_impl(
    blocks, page_num, column_boundaries
):
    element_ids = []
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text.isdigit() and len(text) >= 4:
                        x0 = span["bbox"][0]

                        # Determine column number
                        column = 0  # Default to first column
                        if len(column_boundaries) != 0:
                            for col, boundaries in column_boundaries.items():
                                if boundaries[0] < x0 < boundaries[1]:
                                    column = col
                                    break

                        element_ids.append(
                            {
                                "number": text,
                                "bbox": span["bbox"],  # (x0, y0, x1, y1)
                                "page": page_num,
                                "column": column,
                            }
                        )

    return element_ids


def extract_element_ids_and_positions(doc, page_numbers):
    element_ids = []
    column_boundaries = {}

    for page_num in range(page_numbers["first"], page_numbers["last"] + 1):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        # First pass: determine column boundaries by analyzing x-coordinates
        x_positions = []
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text.isdigit() and len(text) >= 4:
                            x_positions.append(
                                span["bbox"][0]
                            )  # x0 coordinate

        # Determine column boundaries if we have enough data points
        page_column_boundaries = {}
        if x_positions:
            column = 0
            prev_boundary = 0
            # Sort x positions and find natural groupings
            x_positions.sort()

            # Simple approach: find gaps between x positions
            prev_x = x_positions[0]
            for x in x_positions[1:]:
                if x - prev_x > 50:  # Threshold for new column
                    curr_boundary = (prev_x + x) / 2
                    page_column_boundaries[column] = (
                        prev_boundary,
                        curr_boundary,
                    )
                    column += 1
                    prev_boundary = curr_boundary
                prev_x = x
            # Add last column boundary
            page_column_boundaries[column] = (prev_boundary, page.rect.width)

        # Second pass: assign column numbers based on boundaries
        element_ids.extend(
            extract_element_ids_and_positions_impl(
                blocks, page_num, page_column_boundaries
            )
        )

        column_boundaries[page_num] = page_column_boundaries

    # Sort pieces by page, column and vertical position
    element_ids.sort(key=lambda x: (x["page"], x["column"], x["bbox"][1]))
    return element_ids, column_boundaries


def extract_images_and_positions(doc, page_numbers, rejected_images_dir):
    images = []

    for page_num in range(page_numbers["first"], page_numbers["last"] + 1):
        page = doc[page_num]

        # Get all images on the page
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)

            if base_image["width"] < 10 or base_image["height"] < 10:
                # Skip small images
                continue

            # Get image bytes
            image_bytes = base_image["image"]

            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            # Convert to numpy array
            numpy_array = np.array(pil_image)

            # Get the image rectangle (bbox) from the page
            bbox = None
            for rect in page.get_image_rects(xref):
                # Just take the first occurrence's bbox
                bbox = rect
                break

            # Only add images we can locate on the page that aren't solid-color bars
            if bbox and is_complex_image(numpy_array):
                images.append(
                    {
                        "image_data": numpy_array,
                        "ext": base_image["ext"],
                        "bbox": bbox,
                        "page": page_num,
                        "width": base_image["width"],
                        "height": base_image["height"],
                        "xref": xref,
                        "processed": False,
                    }
                )
            else:
                logger.debug(f"Skipping image {xref} on page {page_num}")
                if rejected_images_dir:
                    save_images(
                        rejected_images_dir,
                        [
                            {
                                "image_data": numpy_array,
                                "ext": base_image["ext"],
                                "xref": xref,
                            }
                        ],
                    )

    # Sort images by page and vertical position
    images.sort(key=lambda x: (x["page"], x["bbox"][1]))
    return images


def match_element_ids_to_images(
    element_ids, column_boundaries, images, output_dir
):
    def find_nearest_element_id(element_ids, y):
        try:
            return next(id for id in element_ids if id["bbox"][1] > y)
        except StopIteration:
            return None  # Return None if no item is higher than the threshold

    matches = {}
    potential_matches = {}
    unmatched_images = []

    # Process each page separately
    page_numbers = set(id["page"] for id in element_ids)
    for page_num in page_numbers:
        page_element_ids = [p for p in element_ids if p["page"] == page_num]
        page_images = [i for i in images if i["page"] == page_num]

        # Group element IDs by column
        column_elements = {}
        for element_id in page_element_ids:
            col = element_id["column"]
            if col not in column_elements:
                column_elements[col] = []
            column_elements[col].append(element_id)

        # For each column, match images to the element ID below them
        for col, elements in column_elements.items():
            # Sort elements by vertical position (top to bottom)
            elements.sort(key=lambda x: x["bbox"][1])

            # Get images in this column
            col_images = [
                img
                for img in page_images
                if not img["processed"]
                and column_boundaries[page_num][col][0]
                <= img["bbox"][0]
                < column_boundaries[page_num][col][1]
            ]

            # Sort images by vertical position (top to bottom)
            col_images.sort(key=lambda x: x["bbox"][1])
            # Match each image to the first element ID below it
            for img in col_images:
                img["processed"] = True
                # Match this image to the corresponding element ID
                element_id = find_nearest_element_id(elements, img["bbox"][3])

                if not element_id:
                    # This image doesn't have a corresponding element ID below it
                    # Try and find the ID below the top of the image
                    element_id = find_nearest_element_id(
                        elements, img["bbox"][1]
                    )

                    if not element_id:
                        # We still couldn't find a matching element ID
                        # Save the image as unmatched
                        unmatched_images.append(img)
                        continue

                    # We found an element ID below the top of the image
                    # We'll save this image as a potential match for now
                    if element_id["number"] not in potential_matches:
                        potential_matches[element_id["number"]] = []
                    potential_matches[element_id["number"]].append(img)
                    continue

                # Record the match
                if element_id["number"] not in matches:
                    matches[element_id["number"]] = []
                matches[element_id["number"]].append(img)

            # Check for element IDs without matching images
            if len(matches) < len(elements):
                for j in range(len(col_images), len(elements)):
                    logger.warning(
                        f"No images found for piece {elements[j]['number']}"
                    )

    # Add potential matches to the matched images
    # if no other images were found for the piece
    for element_id, img_refs in potential_matches.items():
        if element_id not in matches:
            matches[element_id] = img_refs
        else:
            unmatched_images.extend(img_refs)

    # Add any remaining unprocessed images
    unmatched_images.extend(
        [image for image in images if not image["processed"]]
    )

    if output_dir:
        # Only save the images if the output directory is provided

        # Create the output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save matched images
        for element_id, images in matches.items():
            for i, image in enumerate(images):
                # Save the image
                filename = f"{element_id}_{i}_{image["xref"]}.{image['ext']}"
                img = Image.fromarray(image["image_data"])
                img.save(output_dir / filename)

        # Save unmatched images
        format = "unmatched_P{page}_y{bbox[1]}_xref{xref}.{ext}"
        save_images(output_dir, unmatched_images, filename_format=format)

    return matches, unmatched_images


def extract_parts_list_from_pdf(
    doc,
    parts_list_pages,
    identified_set_pieces_dir=None,
    parts_list_images_dir=None,
    rejected_images_dir=None,
):
    logger.info("Extracting element IDs")
    element_ids, column_boundaries = extract_element_ids_and_positions(
        doc, parts_list_pages
    )
    logger.info(f"Found {len(element_ids)} element IDs")

    logger.info("Extracting images...")
    images = extract_images_and_positions(
        doc, parts_list_pages, rejected_images_dir
    )
    logger.info(f"Found {len(images)} images")

    if parts_list_images_dir:
        # Save all extracted images in the page before identifying
        logger.info(f"Saving all extracted images to {parts_list_images_dir}")

        save_images(parts_list_images_dir, images)

    logger.info("Matching element IDs to images...")
    matched, unmatched = match_element_ids_to_images(
        element_ids, column_boundaries, images, identified_set_pieces_dir
    )

    logger.info("Results:")
    logger.info(f"pieces with matches: {len(matched)}")
    logger.info(f"Unmatched images: {len(unmatched)}")

    return matched, unmatched


def main():
    import fitz
    from logger import setup_logging

    setup_logging(logging.INFO)
    # Test the code
    manual = "6285327"
    set_num = "75957"
    booklet_num = "1"
    pdf_path = Path(f"data/training/manuals/{manual}.pdf")
    doc = fitz.open(pdf_path)
    piece_renders_dir = Path(
        f"data/processed_booklets/{set_num}_{booklet_num}/piece_renders"
    )
    page_numbers = {"first": 81, "last": 82}  # Pages 82-83 in the PDF

    matched, unmatched = extract_parts_list_from_pdf(
        doc, page_numbers, piece_renders_dir
    )

    # Log unmatched images info
    if unmatched:
        unmatched.sort(key=lambda x: x["xref"])
        logger.info("Unmatched images details:")
        for img in unmatched:
            logger.info(
                f"xref: {img['xref']}, Page: {img['page']}, Position: {img['bbox']}, Size: {img['width']}x{img['height']}"
            )

    doc.close()


if __name__ == "__main__":
    main()
