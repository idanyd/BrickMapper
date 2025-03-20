from pathlib import Path
from typing import List, Dict, Any
import logging
from PIL import Image
from step_detector import StepDetector
from typing import Optional
from utils.pdf_image_extractor import (
    extract_pages_as_images_generator,
    extract_pieces_from_step,
)
from utils.pdf_parts_list_extractor import extract_parts_list_from_pdf
from piece_matcher import PieceMatcher
import io
import fitz
from collections import Counter


class PDFProcessor:
    """Handles extraction and processing of LEGO instruction manual PDFs."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process_manual(
        self,
        doc: fitz.Document,
        page_images_path: Path = None,
        step_pieces_path: Path = None,
        steps_path: Path = None,
    ) -> Dict[Any, Dict[Any, List[Dict]]]:
        """Process a single PDF manual and extract steps."""

        try:
            all_step_pieces = []
            detector = StepDetector()

            for page_img in extract_pages_as_images_generator(doc):
                img = page_img["image_data"]
                page_num = page_img["page_num"]
                if page_images_path:
                    self.logger.info(f"Saving all pages to {page_images_path}")
                    # Create output directory if it doesn't exist
                    page_images_path.mkdir(parents=True, exist_ok=True)
                    self._save_page_image(img, page_num, page_images_path)

                page_step_pieces = []
                # Process each detected step
                try:
                    self.logger.info(f"Detecting steps for page {page_num}")
                    steps = detector.detect_steps(img)
                except Exception as e:
                    self.logger.error(f"Error detecting steps: {str(e)}")
                    return None

                for step_number, step_info in steps.items():
                    # Extract all piece images from the step box if possible
                    step_piece_images = self._extract_step_pieces(
                        doc, page_num - 1, step_info
                    )
                    if step_piece_images and step_info.size:
                        page_step_pieces = [
                            {
                                "img": piece_img,
                                "page": page_num,
                                "step": step_number,
                                "piece": i,
                            }
                            for i, piece_img in enumerate(step_piece_images)
                        ]

                        # Add to list of all step pieces
                        all_step_pieces.extend(page_step_pieces)

                        # Save piece images
                        if step_pieces_path:
                            image_paths = []
                            # Create output directory if it doesn't exist
                            step_pieces_path.mkdir(parents=True, exist_ok=True)
                            for i, piece in enumerate(page_step_pieces):
                                img = Image.open(io.BytesIO(piece["img"]))
                                image_paths.append(
                                    str(
                                        self._save_piece_image(
                                            img,
                                            step_pieces_path,
                                            page_num,
                                            step_number,
                                            i,
                                        )
                                    )
                                )

                    else:
                        # Couldn't find distinct piece images, so process the entire step image
                        step_image = self._extract_step_image(img, step_info)

                        if steps_path and step_image and step_info.size:
                            # Create output directory if it doesn't exist
                            steps_path.mkdir(parents=True, exist_ok=True)
                            # Save step image
                            self._save_step_image(
                                step_image,
                                steps_path,
                                page_num,
                                step_number,
                            )

            return all_step_pieces

        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            raise e

    def _extract_step_pieces(self, doc: Path, page_num: int, step_info: Dict):
        """Extract renders of pieces from the step box."""
        step_images = extract_pieces_from_step(doc, page_num, step_info)
        return step_images

    def _extract_step_image(
        self, page_image: Image.Image, bbox: list[float, float, float, float]
    ) -> Optional[Image.Image]:
        """Extract and process a single step image from the page."""
        try:
            return page_image.crop(bbox)
        except Exception as e:
            self.logger.error(f"Error extracting step image: {str(e)}")
            return None

    def _save_image(self, image: Image.Image, filename: str, output_dir: Path):
        output_path = output_dir / filename
        image.save(output_path)
        return output_path

    def _save_page_image(
        self, image: Image.Image, page_num: int, output_dir: Path
    ):
        """Save a page image with standardized naming."""
        filename = f"page_{page_num:03d}.jpg"
        return self._save_image(image, filename, output_dir)

    def _save_step_image(
        self,
        image: Image.Image,
        output_dir: Path,
        page_num: int,
        step_num: int,
    ) -> Path:
        """Save a step image with standardized naming."""
        filename = f"page_{page_num:03d}_step_{step_num:03d}.jpg"
        return self._save_image(image, filename, output_dir)

    def _save_piece_image(
        self,
        image: Image.Image,
        output_dir: Path,
        page_num: int,
        step_num: int,
        piece_num: int,
    ):
        """Save a piece image with standardized naming."""
        filename = f"page_{page_num:03d}_step_{step_num:03d}_piece_{piece_num:03d}.jpg"
        return self._save_image(image, filename, output_dir)

    def _analyze_page_structure(self, page):
        """Analyze the structure of a page to determine if it's a parts list"""
        # Extract text blocks
        text_blocks = page.get_text("dict")["blocks"]

        potential_element_ids = []

        # Process text blocks to get positions
        text_boxes = []
        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text_boxes.append({"x0": span["bbox"][0]})

        # Extract all text blocks from the page

        blocks = page.get_text("dict")["blocks"]

        # Find all potential element IDs on the page
        x_positions = []
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        # Count numbers that look like element IDs (more than 4 digits)
                        if text.isdigit() and len(text) >= 4:
                            potential_element_ids.append(text)
                            x_positions.append(
                                round(span["bbox"][0])
                            )  # Take the top left corner

        # Analyze column structure
        if x_positions:

            # Count occurrences of each position to detect grid patterns
            x_counts = Counter(x_positions)

            # Columns would have multiple elements (>=3) sharing the same x position
            num_cols = sum(1 for count in x_counts.values() if count >= 3)

            return {
                "element_ids_count": len(potential_element_ids),
                "num_cols": num_cols,
            }
        else:
            return {
                "element_ids_count": len(potential_element_ids),
                "num_cols": 0,
            }

    def _is_parts_list_page(self, page):
        """Determine if a page is likely a parts list based on the analysis"""
        analysis = self._analyze_page_structure(page)

        has_many_ids = analysis["element_ids_count"] > 10
        has_columns = analysis["num_cols"] >= 3

        self.logger.debug(f"Element IDs: {analysis['element_ids_count']}")
        self.logger.debug(f"Number of columns: {analysis['num_cols']}")
        self.logger.debug(f"Is parts list: {has_many_ids and has_columns}")

        return has_many_ids and has_columns

    def find_parts_list_pages(self, doc):
        """Find parts list pages in a LEGO manual PDF"""
        try:
            total_pages = len(doc)

            # Start from the end and work backwards to find parts list pages
            parts_list_pages = []
            in_parts_list_section = False

            for page_num in range(total_pages - 1, -1, -1):
                self.logger.debug(
                    f"Analyzing page {page_num + 1}/{total_pages}..."
                )
                is_parts = self._is_parts_list_page(doc[page_num])

                if in_parts_list_section:
                    if not is_parts:
                        # End of parts list section going backwards
                        # Mark the next page as the first page
                        parts_list_pages.append(page_num + 1)
                        break
                else:
                    if is_parts:
                        # Found the last page of parts list
                        in_parts_list_section = True
                        parts_list_pages.append(page_num)

            # Sort pages in ascending order
            parts_list_pages.sort()

            if parts_list_pages:
                self.logger.debug(
                    f"First parts list page: {parts_list_pages[0]+1}"
                )
                self.logger.debug(
                    f"Last parts list page: {parts_list_pages[1]+1}"
                )
            else:
                self.logger.warning("No parts list pages found.")

            return parts_list_pages

        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            return None


def main():
    from utils.logger import setup_logging

    setup_logging(logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Set up the test data
    data_dir = Path("data")
    manuals_path = data_dir / "training" / "manuals"

    pdf_processor = PDFProcessor()
    matchers = {}

    manuals = {
        "6497660": ("31147", "3"),
    }
    all_sets_set_pieces = {}
    documents = {}
    for manual in manuals:
        set_num, booklet_num = manuals[manual][0], manuals[manual][1]
        # Set up directories
        pdf_path = manuals_path / f"{manual}.pdf"
        documents[manual] = fitz.open(pdf_path)

        # You should only set these directories if you want to save the intermediate images
        # booklet_images_dir = data_dir / "processed_booklets" / manual
        # set_pieces_dir = booklet_images_dir / "set pieces"
        # rejects_dir = booklet_images_dir / "rejected inventory images"

        parts_list_pages = pdf_processor.find_parts_list_pages(
            documents[manual]
        )
        if not parts_list_pages:
            # Couldn't find a parts list in the manual.
            # We'll check later if other manuals from the same set have one
            continue

        if set_num not in all_sets_set_pieces:
            all_sets_set_pieces[set_num] = {}

        all_sets_set_pieces[set_num][booklet_num], _ = (
            extract_parts_list_from_pdf(documents[manual], parts_list_pages)
        )

    # Match each manual with corresponding set pieces
    for manual in manuals:
        set_num, booklet_num = manuals[manual][0], manuals[manual][1]

        # Set up directories

        # You should only set these directories if you want to save the intermediate images
        # booklet_images_dir = data_dir / "processed_booklets" / manual
        # step_pieces_dir = booklet_images_dir / "step pieces"
        # steps_dir = booklet_images_dir / "steps"
        # page_images_dir = booklet_images_dir / "pages"

        if set_num in all_sets_set_pieces:
            if booklet_num in all_sets_set_pieces[set_num]:
                set_pieces = all_sets_set_pieces[set_num][booklet_num]
            else:
                # The parts list wasn't found for this manual.
                # Try to use the parts list from another manual in the same set
                _, set_pieces = next(
                    iter(all_sets_set_pieces[set_num].items())
                )
        else:
            logger.warning(
                f"No parts list found for {manual} in set {set_num}"
            )
            continue

        # Extract steps
        logger.info(f"Processing manual {manual}...")
        try:
            all_step_pieces = pdf_processor.process_manual(documents[manual])
        except Exception as e:
            logger.error(f"Error processing manual {manual}: {e}")
            continue

        matcher = PieceMatcher()

        matcher.add_step_pieces(all_step_pieces)
        matcher.add_set_pieces(set_pieces)

        matcher.match_pieces()

        matchers[manual] = matcher

    for doc in documents.values():
        doc.close()


if __name__ == "__main__":
    main()
