from pathlib import Path
import cv2
from skimage.metrics import structural_similarity as ssim
import re
import pandas as pd
from io import BytesIO
from PIL import Image
import numpy as np
from IPython.display import display, HTML
import base64
import logging


class PieceMatcher:
    """Class for matching LEGO pieces from instruction steps to set piece renders."""

    def __init__(self):
        """Initialize the PieceMatcher class."""
        # Store pieces and renders in memory
        self.step_pieces = (
            []
        )  # Format: [{"img": img_array, "page": page_num, "step": step_num, "piece": piece_num, "path": path}]
        self.set_pieces = (
            {}
        )  # Format: {element_id: [{"image_data": img_array, "path": path}]}
        self.matched = {}  # Matching results
        self.unmatched = {}  # Unmatched results

        self.logger = logging.getLogger(__name__)

    def load_step_pieces_from_directory(self, pieces_dir):
        """
        Load all step piece images from the specified directory.

        Parameters:
        -----------
        pieces_dir : Path or str
            Directory containing the step piece images
        """
        pieces_dir = Path(pieces_dir)

        # Pattern to extract page, step, and piece numbers from filenames
        pattern = r"page_(\d+)_step_(\d+)_piece_(\d+)"

        for file_path in pieces_dir.glob("*.png") or pieces_dir.glob("*.jpg"):
            match = re.search(pattern, file_path.name)
            if match:
                page_num = int(match.group(1))
                step_num = int(match.group(2))
                piece_num = int(match.group(3))

                # Load the image
                img = cv2.imread(str(file_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Store the image and its metadata
                self.step_pieces.append(
                    {
                        "img": img,
                        "page": page_num,
                        "step": step_num,
                        "piece": piece_num,
                        "path": file_path,
                    }
                )

        self.logger.info(
            f"Loaded {len(self.step_pieces)} step pieces from {pieces_dir}"
        )

    def load_set_pieces_from_directory(self, renders_dir):
        """
        Load all set piece render images from the specified directory.

        Parameters:
        -----------
        renders_dir : Path or str
            Directory containing the set piece render images
        """
        renders_dir = Path(renders_dir)

        for file_path in renders_dir.glob("*.png") or renders_dir.glob(
            "*.jpg"
        ):
            # Extract element ID from filename
            element_id = self._get_element_id(file_path.name)
            if element_id:
                # Load the image
                img = cv2.imread(str(file_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Store the image and its path
                if element_id not in self.set_pieces:
                    self.set_pieces[element_id] = []

                self.set_pieces[element_id].append(
                    {"image_data": img, "path": file_path}
                )

        self.logger.info(
            f"Loaded {len(self.set_pieces)} set pieces from {renders_dir}"
        )

    def add_set_pieces(self, set_pieces):
        """
        Add set piece images to the matcher.

        Parameters:
        -----------
        set_pieces : dict
            Dictionary with element IDs as keys and lists of image data as values
        """
        for element_id, images in set_pieces.items():
            if element_id not in self.set_pieces:
                self.set_pieces[element_id] = []
            self.set_pieces[element_id].extend(images)

    def add_step_pieces(self, step_pieces):
        """
        Add step piece images to the matcher.

        Parameters:
        -----------
        step_pieces : list
            List of dictionaries with image data and metadata
        """
        self.step_pieces.extend(step_pieces)

    def _get_element_id(self, filename):
        """
        Extract element ID from piece_renders filename.

        Parameters:
        -----------
        filename : str
            Filename to extract element ID from

        Returns:
        --------
        str or None
            Element ID if found, None otherwise
        """
        match = re.match(r"(\d+)_\d*\.(jp(e?)g|png)", filename)
        if match:
            return match.group(1)
        return None

    def _resize_image(self, img_bytes, target_size=(200, 200)):
        """
        Resize an image to a standard size.

        Parameters:
        -----------
        img_bytes : bytes
            Image data in bytes format
        target_size : tuple, optional
            Target size (width, height)

        Returns:
        --------
        numpy.ndarray
            Resized image
        """
        try:
            # Convert to PIL Image
            image_pil = Image.open(BytesIO(img_bytes))

            # Resize the image
            resized_image = image_pil.resize(
                target_size, Image.Resampling.LANCZOS
            )

            # Convert to RGB if it's not already
            if resized_image.mode != "RGB":
                resized_image = resized_image.convert("RGB")

            # Convert back to bytes array
            # img_byte_arr = BytesIO()
            # resized_image.save(img_byte_arr, format="PNG")

            # return img_byte_arr.getvalue()
            return resized_image
        except Exception as e:
            self.logger.error(f"Error resizing image: {e}")
            return None

    def _compare_images(self, img1, img2):
        """
        Compare two images using SSIM.

        Parameters:
        -----------
        img1 : numpy.ndarray
            First image
        img2 : numpy.ndarray
            Second image

        Returns:
        --------
        float
            Similarity score between 0.0 and 1.0
        """
        score, _ = ssim(
            np.array(img1), np.array(img2), channel_axis=2, full=True
        )
        return score

    def match_pieces(self, similarity_threshold=0.6):
        """
        Match step pieces to set pieces based on image similarity.

        Parameters:
        -----------
        similarity_threshold : float, optional
            Minimum similarity score to consider a match (0.0 to 1.0)

        Returns:
        --------
        tuple
            Dictionaries with matching/unmatched results
        """
        # Reset results
        self.matched = {}
        self.unmatched = {}

        # Resize each set piece image
        resized_set_pieces = {}
        for element_id, render_imgs in self.set_pieces.items():
            resized_set_pieces[element_id] = [
                self._resize_image(render_img["image_data"])
                for render_img in render_imgs
                if render_img["image_data"] is not None
            ]

        # Process each piece image
        for piece in self.step_pieces:
            piece_image = self._resize_image(piece["img"])
            page_num = piece["page"]
            step_num = piece["step"]
            piece_num = piece["piece"]

            if piece_image is None:
                continue

            # Compare with each render image
            best_match = (None, -1)  # (element_id, similarity_score)
            for element_id, render_imgs in resized_set_pieces.items():
                for render_img in render_imgs:
                    # Skip if render image is None
                    if render_img is None:
                        continue

                    # Calculate similarity
                    similarity = self._compare_images(piece_image, render_img)
                    if similarity > best_match[1]:
                        best_match = (element_id, similarity)

            # Choose the appropriate dictionary based on the similarity score
            d = (
                self.matched
                if best_match[1] > similarity_threshold
                else self.unmatched
            )

            element_id = best_match[0]
            if element_id not in d:
                d[element_id] = {}
            if page_num not in d[element_id]:
                d[element_id][page_num] = []

            d[element_id][page_num].append(
                {
                    "step": step_num,
                    "piece": piece_num,
                    "similarity": best_match[1],
                }
            )

        return self.matched, self.unmatched

    def _resize_image_for_display(self, img_bytes, max_size=(100, 100)):
        """Resize image and convert to base64 for HTML display"""
        img_pil = Image.open(BytesIO(img_bytes))
        img_pil.thumbnail(max_size)
        img_bio = BytesIO()
        img_pil.save(img_bio, format="PNG")
        img_b64 = base64.b64encode(img_bio.getvalue()).decode("utf-8")
        return f'<img src="data:image/png;base64,{img_b64}"/>'

    def create_comparison_dataframe(self, compare_unmatched=False):
        """
        Create a Pandas DataFrame showing matched/unmatched pieces and their similarity scores
        using data from the matched/unmatched dictionaries and the stored images.

        Returns:
        --------
        pd.DataFrame
            DataFrame with comparison data and embedded images
        """
        d = self.unmatched if compare_unmatched else self.matched
        if not d:
            self.logger.info("No results found. Run match_pieces() first.")
            return pd.DataFrame()

        # Create lists to store data for DataFrame
        data_rows = []

        for element_id, pages in d.items():
            # Find the corresponding render image
            if element_id not in self.set_pieces:
                continue

            render_data = self.set_pieces[element_id][
                0
            ]  # Use the first render image
            render_img = render_data["image_data"]
            render_path = (
                render_data["path"] if "path" in render_data else None
            )

            for page_num, steps_data in pages.items():
                for step_info in steps_data:
                    step_num = step_info["step"]
                    similarity = step_info["similarity"]
                    piece_num = step_info["piece"]

                    # Find the corresponding piece image
                    piece_data = None
                    for piece in self.step_pieces:
                        if (
                            piece["page"] == page_num
                            and piece["step"] == step_num
                            and piece["piece"] == piece_num
                        ):
                            piece_data = piece
                            break

                    if piece_data is None:
                        continue

                    piece_img = piece_data["img"]
                    piece_path = (
                        piece_data["path"] if "path" in piece_data else None
                    )

                    # Create HTML for images
                    piece_html = self._resize_image_for_display(piece_img)
                    render_html = self._resize_image_for_display(render_img)

                    # Add row to data
                    data_rows.append(
                        {
                            "Element ID": element_id,
                            "Page": page_num,
                            "Step": step_num,
                            "Piece Image": piece_html,
                            "Render Image": render_html,
                            "Similarity Score": f"{similarity:.3f}",
                            "Piece Filename": (
                                piece_path.name if piece_path else ""
                            ),
                            "Render Filename": (
                                render_path.name if render_path else ""
                            ),
                        }
                    )

        # Create DataFrame
        df = pd.DataFrame(data_rows)

        return df

    def display_comparison_dataframe(self, df=None, compare_unmatched=False):
        """
        Display the comparison DataFrame with properly rendered images in a Jupyter notebook

        Parameters:
        -----------
        df : pd.DataFrame, optional
            DataFrame created by create_comparison_dataframe. If None, creates a new one.

        Returns:
        --------
        pd.DataFrame
            DataFrame with comparison data (without HTML)
        """
        if df is None:
            df = self.create_comparison_dataframe(compare_unmatched)

        if df.empty:
            self.logger.info("No data to display.")
            return df

        # Display the DataFrame with HTML rendering enabled
        display(HTML(df.to_html(escape=False)))

        # Return a regular DataFrame without HTML for further processing
        # Create a copy without the HTML content
        clean_df = df.copy()
        if "Piece Image" in clean_df.columns:
            clean_df["Piece Image"] = "[IMAGE]"
        if "Render Image" in clean_df.columns:
            clean_df["Render Image"] = "[IMAGE]"

        return clean_df


def main():
    """Example usage of the PieceMatcher class."""
    pieces_dir = Path("data/processed_booklets/31147_2/pieces")
    renders_dir = Path("data/processed_booklets/31147_3/piece_renders")

    # Initialize the matcher
    matcher = PieceMatcher()

    # Load pieces and renders
    matcher.load_step_pieces_from_directory(pieces_dir)
    matcher.load_set_pieces_from_directory(renders_dir)

    # Match pieces
    matched, unmatched = matcher.match_pieces(similarity_threshold=0.6)

    # Create and display comparison dataframe
    df = matcher.create_comparison_dataframe()
    matcher.display_comparison_dataframe(df)


if __name__ == "__main__":
    main()
