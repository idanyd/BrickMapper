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
from concurrent.futures import ProcessPoolExecutor
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _process_piece_two_pass(args):
    """
    Process a single piece using two-pass matching:
    1. Quick filtering using color histograms
    2. Detailed comparison using SSIM for top matches
    """
    (
        piece,
        processed_set_pieces,
        quick_threshold,
        final_threshold,
        num_top_matches,
    ) = args

    # First pass: Compare color histograms
    piece_hist = PieceMatcher.extract_color_histogram(piece["img"])

    # Store histogram comparisons
    hist_scores = []
    best_first_pass = None
    best_hist_score = 0

    for element_id, renders in processed_set_pieces.items():
        for render_idx, render_data in enumerate(renders):
            # Compare histograms using correlation
            hist_score = cv2.compareHist(
                piece_hist,
                render_data["histogram"],  # Pre-computed histogram
                method=cv2.HISTCMP_CORREL,
            )

            candidate = {
                "element_id": element_id,
                "render_idx": render_idx,
                "score": hist_score,
            }
            # Store candidates that pass the quick threshold
            if hist_score > quick_threshold:
                hist_scores.append(candidate)
            # Store the best histogram match for later comparison
            if hist_score > best_hist_score:
                best_hist_score = hist_score
                best_first_pass = candidate

    if not hist_scores:
        # No matches found in the first passץ Return the best histogram match
        # even if it doesn't pass the quick threshold
        return {
            "piece": piece,
            "element_id": best_first_pass["element_id"],
            "final_similarity": 0,  # Failed quick threshold
            "hist_similarity": best_first_pass["score"],
        }

    # Sort by histogram similarity and get top matches
    hist_scores.sort(key=lambda x: x["score"], reverse=True)
    top_matches = hist_scores[:num_top_matches]

    # Second pass: Calculate SSIM for top matches
    ssim_scores = []
    best_second_pass = None
    best_ssim_score = 0
    for match in top_matches:
        render_data = processed_set_pieces[match["element_id"]][
            match["render_idx"]
        ]

        # Calculate SSIM
        ssim_score, _ = ssim(
            PieceMatcher._resize_image(piece["img"]),
            render_data["image"],
            channel_axis=2,
            full=True,
        )

        candidate = {
            "element_id": match["element_id"],
            "final_similarity": ssim_score,
            "hist_similarity": match["score"],
        }
        # Store candidates that pass the final threshold
        if ssim_score > final_threshold:
            ssim_scores.append(candidate)
        # Store the best SSIM match for later comparison
        if ssim_score > best_ssim_score:
            best_ssim_score = ssim_score
            best_second_pass = candidate

    # If no matches pass the final threshold, return the best SSIM score
    # Else, return the best match that passed both thresholds

    best_match = (
        best_second_pass
        if not ssim_scores
        else max(ssim_scores, key=lambda x: x["final_similarity"])
    )
    return {
        "piece": piece,
        "element_id": best_match["element_id"],
        "final_similarity": best_match["final_similarity"],
        "hist_similarity": best_match["hist_similarity"],
    }


def _remove_background_simple(img, tolerance=20, debug=False):
    """
    Remove the background from an image by identifying the background color
    from the top-left pixel and replacing all pixels of that color with white.

    Parameters:
    -----------
    img : numpy.ndarray
        Input image in RGB format
    debug : bool
        If True, display the mask and the resulting image for debugging

    Returns:
    --------
    numpy.ndarray
        Image with background replaced by white
    """
    # Get background color from the top-left pixel
    bg_color = img[0, 0].astype(int)

    # Create masks for each channel separately
    masks = []
    for channel in range(3):  # RGB channels
        channel_mask = (
            img[:, :, channel] >= max(0, bg_color[channel] - tolerance)
        ) & (img[:, :, channel] <= min(255, bg_color[channel] + tolerance))
        masks.append(channel_mask)

    # Combine masks - a pixel is background only if all channels are within tolerance
    final_mask = masks[0] & masks[1] & masks[2]

    # Create output image (copy of input)
    result = img.copy()

    # Set all background pixels to white
    result[final_mask > 0] = [255, 255, 255]

    if debug:
        # Visualize the mask and the result
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(132)
        plt.imshow(final_mask, cmap="gray")
        plt.title("Background Mask")
        plt.axis("off")

        plt.subplot(133)
        plt.imshow(result)
        plt.title("Image with Background Removed")
        plt.axis("off")

        plt.show()

    return result


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

    def load_step_pieces_from_directory(self, pieces_dir):
        """
        Load all step piece images from the specified directory.

        Parameters:
        -----------
        pieces_dir : Path or str
            Directory containing the step piece images
        """

        # Pattern to extract page, step, and piece numbers from filenames
        pattern = r"page_(\d+)_step_(\d+)_piece_(\d+)"

        for file_path in pieces_dir.glob("*"):
            match = re.search(pattern, file_path.name)
            if match:
                page_num = int(match.group(1))
                step_num = int(match.group(2))
                piece_num = int(match.group(3))

                # Load the image
                img = cv2.imread(str(file_path))
                if img is None:
                    continue

                # Store the image and its metadata
                self.step_pieces.append(
                    {
                        "img": img,
                        "page": page_num,
                        "step": step_num,
                        "piece": piece_num,
                    }
                )

        logger.info(
            f"Loaded {len(self.step_pieces)} step pieces from {pieces_dir}"
        )

    def load_set_pieces_from_directory(self, set_pieces_dir):
        """
        Load all set piece render images from the specified directory.

        Parameters:
        -----------
        renders_dir : Path or str
            Directory containing the set piece render images
        """

        for file_path in set_pieces_dir.glob("*.png") or set_pieces_dir.glob(
            "*.jpg"
        ):
            # Extract element ID from filename
            element_id = self._get_element_id(file_path.name)
            if element_id:
                # Load the image
                img = cv2.imread(str(file_path))
                if img is None:
                    continue

                # Store the image and its path
                if element_id not in self.set_pieces:
                    self.set_pieces[element_id] = []

                self.set_pieces[element_id].append(
                    {"image_data": img, "path": file_path}
                )

        logger.info(
            f"Loaded {len(self.set_pieces)} set pieces from {set_pieces_dir}"
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
        match = re.match(r"(\d+)_\d_\d*\.(jp(e?)g|png)", filename)
        if match:
            return match.group(1)
        return None

    def match_pieces(
        self,
        hist_threshold=0.9,
        similarity_threshold=0.6,
        n_workers=4,
        num_top_matches=15,
    ):
        """
        Match step pieces to set pieces using color histograms and SSIM.
        """
        # Reset results
        self.matched = {}
        self.unmatched = {}

        # Pre-process set pieces (compute histograms)
        processed_set_pieces = {}
        for element_id, renders in self.set_pieces.items():
            processed_set_pieces[element_id] = []
            for render in renders:
                hist = self.extract_color_histogram(render["image_data"])
                processed_set_pieces[element_id].append(
                    {
                        "image": self._resize_image(render["image_data"]),
                        "histogram": hist,
                    }
                )

        # Process pieces in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for piece in self.step_pieces:
                args = (
                    piece,
                    processed_set_pieces,
                    hist_threshold,  # quick_threshold for histogram matching
                    similarity_threshold,  # final_threshold for SSIM
                    num_top_matches,
                )
                futures.append(executor.submit(_process_piece_two_pass, args))

            # Collect results
            for future in futures:
                try:
                    result = future.result()

                    piece = result["piece"]
                    element_id = result["element_id"]
                    similarity = result["final_similarity"]
                    page_num = piece["page"]
                    step_num = piece["step"]
                    piece_num = piece["piece"]

                    # Choose the appropriate dictionary based on the similarity score
                    d = (
                        self.matched
                        if similarity > similarity_threshold
                        else self.unmatched
                    )

                    if element_id not in d:
                        d[element_id] = {}
                    if page_num not in d[element_id]:
                        d[element_id][page_num] = []

                    d[element_id][page_num].append(
                        {
                            "step": step_num,
                            "piece": piece_num,
                            "similarity": similarity,
                            "hist_similarity": result["hist_similarity"],
                        }
                    )
                except Exception as e:
                    print(f"Error processing piece: {e}")

        return self.matched, self.unmatched

    def _resize_image_for_display(self, img, max_size=(100, 100)):
        """Resize image and convert to base64 for HTML display"""
        try:
            img_pil = Image.fromarray(img)
        except Exception:
            # If the image is not in a format that can be converted directly, use BytesIO
            img_pil = Image.open(BytesIO(img))
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
            logger.info("No results found. Run match_pieces() first.")
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
                    histogram_similarity = step_info["hist_similarity"]
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
                            "Histogram Similarity": f"{histogram_similarity:.3f}",
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
            logger.info("No data to display.")
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

    @staticmethod
    def display_dataframe_statistics(dfs):
        """
        Display statistics for the given DataFrames.

        Parameters:
        -----------
        dfs : dict {name: pd.DataFrame}
            DataFrames to display statistics for
        """

        if not dfs:
            logger.info("No DataFrames provided.")
            return

        summary = {}
        for name, df in dfs.items():
            if df.empty:
                continue
            summary[name] = [
                df.shape[0],
                df["Similarity Score"].astype(float).min(),
                df["Similarity Score"].astype(float).max(),
                f"{df["Similarity Score"].astype(float).mean():.3f}",
                df["Histogram Similarity"].astype(float).min(),
                df["Histogram Similarity"].astype(float).max(),
                f"{df["Histogram Similarity"].astype(float).mean():.3f}",
            ]

        # Display the DataFrame
        # Create the summary DataFrame from the dictionary
        summary_df = pd.DataFrame(
            summary,
            index=[
                "Total Pieces",
                "Min Similarity",
                "Max Similarity",
                "Mean Similarity",
                "Min Histogram Similarity",
                "Max Histogram Similarity",
                "Mean Histogram Similarity",
            ],
        )

        display(summary_df)
        return summary_df

    @staticmethod
    def _resize_image(img, target_size=(200, 200)):
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
            try:
                image_pil = Image.fromarray(img)
            except Exception:
                # If the image is not in a format that can be converted directly, use BytesIO
                image_pil = Image.open(BytesIO(img))

            # Resize the image
            resized_image = image_pil.resize(
                target_size, Image.Resampling.LANCZOS
            )

            # Convert to RGB if it's not already
            if resized_image.mode != "RGB":
                resized_image = resized_image.convert("RGB")

            # return img_byte_arr.getvalue()
            return np.array(resized_image)
            # return resized_image
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return None

    @staticmethod
    def extract_color_histogram(img, bins=32):
        """
        Extract color histogram from an image after removing background.

        Parameters:
        -----------
        img : numpy.ndarray
            Input image in RGB format
        bins : int
            Number of bins per color channel

        Returns:
        --------
        numpy.ndarray
            Flattened histogram array
        """
        # Remove background
        processed_img = _remove_background_simple(img)

        # Calculate histogram for each channel
        hist = cv2.calcHist(
            [processed_img],
            channels=[0, 1, 2],  # RGB channels
            mask=None,  # No mask
            histSize=[bins] * 3,  # Same number of bins for each channel
            ranges=[0, 256] * 3,  # Full range for each channel
        )

        # Normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()
        return hist


def main():
    logging.basicConfig(level=logging.INFO)

    # Code profiling
    from cProfile import Profile
    from pstats import SortKey, Stats
    import time

    st = time.time()
    with Profile() as profile:
        manual = "6236540"
        """Example usage of the PieceMatcher class."""
        step_pieces_dir = Path(f"data/processed_booklets/{manual}/step pieces")
        set_pieces_dir = Path(
            f"data/processed_booklets/{manual}/identified set pieces"
        )

        # Initialize the matcher
        matcher = PieceMatcher()

        # Load pieces and renders
        matcher.load_step_pieces_from_directory(step_pieces_dir)
        matcher.load_set_pieces_from_directory(set_pieces_dir)

        # Match pieces
        matched, unmatched = matcher.match_pieces(
            similarity_threshold=0.6, n_workers=16, num_top_matches=5
        )

        with open("stats.txt", "w") as file:
            stats = (
                Stats(profile, stream=file)
                .strip_dirs()
                .sort_stats(SortKey.CUMULATIVE)
            )
            stats.print_stats()

        # Create and display comparison dataframe
        # df = matcher.create_comparison_dataframe()
        # matcher.display_comparison_dataframe(df)
    et = time.time()
    elapsed_time = et - st
    print("Execution time:", elapsed_time, "seconds")


if __name__ == "__main__":
    main()
