from pathlib import Path
from typing import List, Dict
import logging
from PIL import Image
from step_detector import StepDetector
from typing import Optional
from utils.pdf_image_extractor import extract_pages_as_images


class PDFProcessor:
    """Handles extraction and processing of LEGO instruction manual PDFs."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process_manual(
        self,
        pdf_path: Path,
        processed_booklets_path: Path,
        set_num: str,
        booklet_num: str,
    ) -> List[Dict]:
        """Process a single PDF manual and extract steps."""

        try:
            images_dir = processed_booklets_path / f"{set_num}_{booklet_num}"
            # Extract images from PDF
            extract_pages_as_images(
                pdf_path=pdf_path,
                images_dir=images_dir,
                set_num=set_num,
                booklet_num=booklet_num,
            )

            # Create output directory if it doesn't exist
            output_dir = images_dir / "steps"
            output_dir.mkdir(parents=True, exist_ok=True)

            results = []
            detector = StepDetector()

            # Process each image
            for img_file in sorted(images_dir.glob("*.jpg")):
                img = Image.open(img_file)
                page_num = int(img_file.stem.split("_")[2])
                set_num = int(img_file.stem.split("_")[0])
                booklet_num = int(img_file.stem.split("_")[1])

                # Process each detected step
                try:
                    steps = detector.detect_steps(image_path=img_file)
                except Exception as e:
                    self.logger.error(f"Error detecting steps: {str(e)}")
                    return None

                # Process each detected step
                for step_number, step_info in steps.items():
                    step_image = self._extract_step_image(img, step_info)

                    if step_image and step_info.size:
                        # Save step image
                        image_path = self._save_step_image(
                            step_image,
                            output_dir,
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
            return results

        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise e

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


def main():
    # Test the PDFProcessor
    pdf_processor = PDFProcessor()
    pdf_path = Path("data/training/manuals/6497659.pdf")
    processed_booklets_path = Path("data/processed_booklets")
    set_num = "31147"
    booklet_num = "2"
    pdf_processor.process_manual(
        pdf_path, processed_booklets_path, set_num, booklet_num
    )


if __name__ == "__main__":
    main()
