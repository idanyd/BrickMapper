from pathlib import Path
from typing import List, Dict
import logging
from PIL import Image
from step_detector import StepDetector
from typing import Optional
from utils.pdf_image_extractor import (
    extract_pages_as_images,
    extract_pieces_from_step,
)
from utils.pdf_parts_list_extractor import extract_parts_list_from_pdf
from piece_matcher import match_pieces
import io


class PDFProcessor:
    """Handles extraction and processing of LEGO instruction manual PDFs."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process_manual(
        self,
        parts_list_pages: List[int],
        set_num: str,
        booklet_num: str,
        pdf_path: Path,
        booklet_images_path: Path,
        set_pieces_path: Path = None,
        step_pieces_path: Path = None,
        steps_path: Path = None,
        rejected_images_path: Path = None,
    ) -> Dict[Dict[List[Dict]]]:
        """Process a single PDF manual and extract steps."""

        # Extract the parts list from the manual
        if parts_list_pages:
            matched_set_pieces, _ = extract_parts_list_from_pdf(
                pdf_path,
                parts_list_pages,
                set_pieces_path,
                rejected_images_dir=rejected_images_path,
            )

        else:
            self.logger.error("No parts list pages provided.")
            return None

        try:
            # Extract images from PDF, to be used for step detection
            extract_pages_as_images(
                pdf_path=pdf_path,
                images_dir=booklet_images_path,
                set_num=set_num,
                booklet_num=booklet_num,
            )

            results = []
            all_step_pieces = []
            detector = StepDetector()

            # Process each image
            for img_file in sorted(booklet_images_path.glob("*.jpg")):
                img = Image.open(img_file)
                page_num = int(img_file.stem.split("_")[2])
                page_step_pieces = []
                # Process each detected step
                try:
                    steps = detector.detect_steps(img_file)
                except Exception as e:
                    self.logger.error(f"Error detecting steps: {str(e)}")
                    return None

                for step_number, step_info in steps.items():
                    # Extract all piece images from the step box if possible
                    step_piece_images = self._extract_step_pieces(
                        pdf_path, page_num - 1, step_info
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
                                image = Image.open(io.BytesIO(piece["img"]))
                                image_paths.append(
                                    str(
                                        self._save_piece_image(
                                            image,
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
                            image_path = self._save_step_image(
                                step_image,
                                steps_path,
                                page_num,
                                step_number,
                            )

                        results.append(
                            {
                                "page_number": page_num,
                                "step_number": step_number,
                                "image_path": str(image_path),
                            }
                        )

            return match_pieces(all_step_pieces, matched_set_pieces)

        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise e

    def _extract_step_pieces(
        self, pdf_path: Path, page_num: int, step_info: Dict
    ):
        """Extract renders of pieces from the step box."""
        step_images = extract_pieces_from_step(pdf_path, page_num, step_info)
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

    def _save_step_image(
        self,
        image: Image.Image,
        output_dir: Path,
        page_num: int,
        step_num: int,
    ) -> Path:
        """Save a step image with standardized naming."""
        filename = f"page_{page_num:03d}_step_{step_num:03d}.jpg"
        output_path = output_dir / filename
        image.save(output_path)
        return output_path

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
        output_path = output_dir / filename
        image.save(output_path)
        return output_path


def main():
    import csv
    from utils.logger import setup_logging

    setup_logging()

    # Set up the test data
    data_dir = Path("data")
    manuals_path = data_dir / "training" / "manuals"
    manuals_to_sets = manuals_path / "manuals_to_sets.csv"

    manual = "6497660"
    parts_list_pages = [37, 38]

    pdf_processor = PDFProcessor()
    with open(manuals_to_sets, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        manuals_dict = {row[0]: [row[1], row[2]] for row in reader}
        set_num = manuals_dict[manual][0]
        booklet_num = manuals_dict[manual][1]

        # Set up directories
        pdf_path = manuals_path / f"{manual}.pdf"
        booklet_images_dir = (
            data_dir / "processed_booklets" / f"{set_num}_{booklet_num}"
        )

        # You should only set these directories if you want to save the intermediate images
        step_pieces_dir = booklet_images_dir / "step pieces"
        steps_dir = booklet_images_dir / "steps"
        set_pieces_dir = booklet_images_dir / "set pieces"
        rejects_dir = booklet_images_dir / "rejected inventory images"

        # Extract steps
        steps = pdf_processor.process_manual(
            parts_list_pages,
            set_num,
            booklet_num,
            pdf_path,
            booklet_images_dir,
            rejected_images_path=rejects_dir,
        )


if __name__ == "__main__":
    main()
