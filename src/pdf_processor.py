import fitz
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from PIL import Image
import io
from src.lego_step_detector import LEGOStepDetector
from typing import Optional


class PDFProcessor:
    """Handles extraction and processing of LEGO instruction manual PDFs."""

    def __init__(self, model_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.step_detector = LEGOStepDetector(model_path)

    def process_manual(self, pdf_path: Path, output_dir: Path) -> List[Dict]:
        """Process a single PDF manual and extract steps."""
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Extract page as image
                pix = page.get_pixmap()
                img_data = pix.tobytes()
                img = Image.open(io.BytesIO(img_data))

                # Detect steps in the page
                steps = self.step_detector.detect_steps(img)

                # Process each detected step
                for step_index, step_info in enumerate(steps):
                    step_image = self._extract_step_image(
                        img, step_info["bbox"]
                    )
                    # Save the image to a file
                    img_path = (
                        output_dir
                        / f"page_{page_num + 1}_step{step_index}.png"
                    )
                    step_image.save(img_path)
                    print(f"Saved image to {img_path}")

                    if step_image and step_info["step_number"]:
                        # Save step image
                        image_path = self._save_step_image(
                            step_image,
                            output_dir,
                            page_num + 1,
                            step_info["step_number"],
                        )

                        results.append(
                            {
                                "page_number": page_num + 1,
                                "step_number": step_info["step_number"],
                                "image_path": str(image_path),
                            }
                        )

            return results

        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise

    def _extract_step_image(
        self, page_image: Image.Image, bbox: Tuple[float, float, float, float]
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
        filename = f"page_{page_num:03d}_step_{step_num:03d}.png"
        output_path = output_dir / filename
        image.save(output_path)
        return output_path
