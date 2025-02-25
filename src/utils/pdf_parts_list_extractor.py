import fitz
import os
from pathlib import Path


def extract_element_ids_and_positions_impl(
    blocks, page_num, column_boundaries
):
    element_ids = []
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text.isdigit() and len(text) >= 6:
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


def extract_element_ids_and_positions(pdf_path, page_numbers):
    doc = fitz.open(pdf_path)
    element_ids = []

    for page_num in page_numbers:
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        # First pass: determine column boundaries by analyzing x-coordinates
        x_positions = []
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text.isdigit() and len(text) >= 6:
                            x_positions.append(
                                span["bbox"][0]
                            )  # x0 coordinate

        # Determine column boundaries if we have enough data points
        column_boundaries = {}
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
                    column_boundaries[column] = (prev_boundary, curr_boundary)
                    column += 1
                    prev_boundary = curr_boundary
                prev_x = x
            # Add last column boundary
            column_boundaries[column] = (prev_boundary, page.rect.width)

        # Second pass: assign column numbers based on boundaries
        element_ids.extend(
            extract_element_ids_and_positions_impl(
                blocks, page_num, column_boundaries
            )
        )

    doc.close()
    # Sort pieces by page, column and vertical position
    element_ids.sort(key=lambda x: (x["page"], x["column"], x["bbox"][1]))
    return element_ids, column_boundaries


def extract_images_and_positions(pdf_path, page_numbers):
    doc = fitz.open(pdf_path)
    images = []

    for page_num in page_numbers:
        page = doc[page_num]

        # Get all images on the page
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)

            # Get the image rectangle (bbox) from the page
            pix = fitz.Pixmap(doc, xref)
            bbox = None

            # Search for this image on the page
            for img_info in page.get_image_info():
                if (
                    img_info["width"] == pix.width
                    and img_info["height"] == pix.height
                ):
                    bbox = img_info["bbox"]
                    break

            if bbox:  # Only add images we can locate on the page
                images.append(
                    {
                        "image_data": base_image["image"],
                        "ext": base_image["ext"],
                        "bbox": bbox,
                        "page": page_num,
                        "width": base_image["width"],
                        "height": base_image["height"],
                        "matched": False,
                    }
                )

    doc.close()
    # Sort images by page and vertical position
    images.sort(key=lambda x: (x["page"], x["bbox"][1]))
    return images


def match_element_ids_to_images(
    element_ids, column_boundaries, images, output_dir
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    matches = {}
    unmatched_images = []

    # Process each page separately
    page_numbers = set(id["page"] for id in element_ids)
    for page_num in page_numbers:
        page_element_ids = [p for p in element_ids if p["page"] == page_num]
        page_images = [i for i in images if i["page"] == page_num]

        # For each element id, find images between this id and the id above it
        for i, element_id in enumerate(page_element_ids):
            first_in_column = (
                i == 0
                or element_id["column"] != page_element_ids[i - 1]["column"]
            )
            current_y = element_id["bbox"][1]
            # If this is the first piece on the page, use top of page as upper bound
            upper_y = (
                0 if first_in_column else page_element_ids[i - 1]["bbox"][1]
            )

            # Find all images in the column between upper_y and current_y
            matching_images = [
                img
                for img in page_images
                if not img["matched"]
                and upper_y <= img["bbox"][1] <= current_y
                and column_boundaries[element_id["column"]][0]
                <= img["bbox"][0]
                < column_boundaries[element_id["column"]][1]
                and img["width"] > 10
                and img["height"] > 10  # Filter out small images
            ]

            if matching_images:
                # Save images and record matches
                for idx, img in enumerate(matching_images):
                    img["matched"] = True
                    filename = (
                        f"piece_{element_id['number']}_{idx}.{img['ext']}"
                    )
                    with open(os.path.join(output_dir, filename), "wb") as f:
                        f.write(img["image_data"])

                    if element_id["number"] not in matches:
                        matches[element_id["number"]] = []
                    matches[element_id["number"]].append(filename)
            else:
                print(
                    f"Warning: No images found for piece {element_id['number']}"
                )

    # Images that weren't matched to any piece
    unmatched_images = [
        img
        for img in images
        if img["width"] > 10 and img["height"] > 10 and img["matched"] == False
    ]

    return matches, unmatched_images


# Test the code
pdf_path = Path("data/training/manuals/6127614.pdf")
output_dir = "piece_renders"
page_numbers = [67, 68]  # Pages 68-69 in the PDF

print("Extracting element IDs...")
element_ids, column_boundaries = extract_element_ids_and_positions(
    pdf_path, page_numbers
)
print(f"Found {len(element_ids)} element IDs")

print("\nExtracting images...")
images = extract_images_and_positions(pdf_path, page_numbers)
print(f"Found {len(images)} images")

print("\nMatching element IDs to images...")
matches, unmatched = match_element_ids_to_images(
    element_ids, column_boundaries, images, output_dir
)

print("\nResults:")
print(f"pieces with matches: {len(matches)}")
print(f"Unmatched images: {len(unmatched)}")

# Print some example matches
print("\nSample matches (first 5 pieces):")
for element_id in list(matches.keys())[:5]:
    print(f"piece {element_id} -> {len(matches[element_id])} images")

# Print unmatched images info
if unmatched:
    print("\nUnmatched images details:")
    for img in unmatched:
        print(
            f"Page {img['page']}, Position: {img['bbox']}, Size: {img['width']}x{img['height']}"
        )
