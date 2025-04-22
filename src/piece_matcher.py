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
from concurrent.futures import ThreadPoolExecutor
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)


# Dataclasses for configuration parameters
@dataclass
class MatchingConfig:
    """Configuration parameters for piece matching"""

    quick_threshold: float
    final_threshold: float
    num_top_matches: int


class HistogramMatch(TypedDict):
    """Type definition for histogram matching results"""

    element_id: str
    set_piece_idx: int
    score: float


class MatchResult(TypedDict):
    """Type definition for final matching results"""

    piece: dict
    element_id: str
    final_similarity: float
    hist_similarity: float


def compare_histograms(
    step_piece_hist: np.ndarray, set_piece_hist: np.ndarray
) -> float:
    """Compare two color histograms using correlation method"""
    return cv2.compareHist(
        step_piece_hist, set_piece_hist, method=cv2.HISTCMP_CORREL
    )


def find_best_histogram_matches(
    piece_hist: np.ndarray, processed_set_pieces: Dict, config: MatchingConfig
) -> tuple[List[HistogramMatch], Optional[HistogramMatch]]:
    """
    First pass: Compare color histograms and find best matches

    Returns:
        Tuple of (list of matches above threshold, best match regardless of threshold)
    """
    hist_scores = []
    best_first_pass = None
    best_hist_score = 0
    logger.debug("Finding best histogram matches...")
    for element_id, pieces in processed_set_pieces.items():
        for set_piece_idx, set_piece_data in enumerate(pieces):
            hist_score = compare_histograms(
                piece_hist, set_piece_data["histogram"]
            )

            candidate = {
                "element_id": element_id,
                "set_piece_idx": set_piece_idx,
                "score": hist_score,
            }

            if hist_score > config.quick_threshold:
                hist_scores.append(candidate)

            if hist_score > best_hist_score:
                best_hist_score = hist_score
                best_first_pass = candidate

    return hist_scores, best_first_pass


def calculate_ssim_score(
    step_piece_img: np.ndarray, set_piece_img: np.ndarray
) -> float:
    """Calculate SSIM score between two images"""
    ssim_score, _ = ssim(
        step_piece_img, set_piece_img, channel_axis=2, full=True
    )
    return ssim_score


def ssim_match_candidates(
    resized_step_piece: np.ndarray,
    candidates: List[dict],
    processed_set_pieces: Dict,
    final_threshold: float,
    use_hist_score: bool = True,
) -> tuple[List[dict], Optional[dict]]:
    """
    Calculate SSIM for a list of candidate matches.

    Args:
        resized_step_piece: The query image (already resized).
        candidates: List of dicts, each with 'element_id', 'set_piece_idx', and optionally 'score'.
        processed_set_pieces: Dict of all set pieces.
        final_threshold: SSIM threshold for a match.
        use_hist_score: If True, use candidate['score'] as hist_similarity; else set to 0.

    Returns:
        (matches_above_threshold, best_match)
    """
    logger.debug("Calculating SSIM for candidates...")

    # Initialize lists to store results
    ssim_scores = []
    best_match = None
    best_ssim_score = 0

    for candidate in candidates:
        element_id = candidate["element_id"]
        set_piece_idx = candidate.get("set_piece_idx", 0)
        set_piece_data = processed_set_pieces[element_id][set_piece_idx]
        ssim_score = calculate_ssim_score(
            resized_step_piece, set_piece_data["image"]
        )

        match_result = {
            "element_id": element_id,
            "final_similarity": ssim_score,
            "hist_similarity": candidate["score"] if use_hist_score else 0,
        }

        if ssim_score > final_threshold:
            ssim_scores.append(match_result)

        if ssim_score > best_ssim_score:
            best_ssim_score = ssim_score
            best_match = match_result

    return ssim_scores, best_match


def find_best_ssim_matches(
    resized_step_piece: np.ndarray,
    top_hist_matches: List[HistogramMatch],
    processed_set_pieces: Dict,
    config: MatchingConfig,
) -> tuple[List[dict], Optional[dict]]:
    """
    Second pass: Calculate SSIM for top histogram matches

    Returns:
        Tuple of (list of matches above threshold, best match regardless of threshold)
    """
    return ssim_match_candidates(
        resized_step_piece,
        top_hist_matches,
        processed_set_pieces,
        config.final_threshold,
        use_hist_score=True,
    )


def fallback_ssim_matching(
    resized_step_piece: np.ndarray,
    processed_set_pieces: Dict,
    config: MatchingConfig,
) -> tuple[List[dict], Optional[dict]]:
    """
    Fallback: Run SSIM matching on all set pieces when no good matches are found
    """
    all_candidates = [
        {"element_id": element_id, "set_piece_idx": idx, "score": 0}
        for element_id, pieces in processed_set_pieces.items()
        for idx in range(len(pieces))
    ]
    return ssim_match_candidates(
        resized_step_piece,
        all_candidates,
        processed_set_pieces,
        config.final_threshold,
        use_hist_score=False,
    )


def _process_piece_two_pass(args: tuple) -> MatchResult:
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

    config = MatchingConfig(
        quick_threshold=quick_threshold,
        final_threshold=final_threshold,
        num_top_matches=num_top_matches,
    )

    logger.debug(
        f"Processing piece: {piece["piece"]}, Page: {piece["page"]}, Step: {piece["step"]}"
    )
    try:
        # First pass: Compare color histograms
        piece_hist = PieceMatcher.extract_color_histogram(piece["img"])
        hist_scores, best_first_pass = find_best_histogram_matches(
            piece_hist, processed_set_pieces, config
        )

        if not hist_scores:
            logger.debug(
                f"No histogram matches found above threshold {config.quick_threshold}"
            )
            return {
                "piece": piece,
                "element_id": best_first_pass["element_id"],
                "final_similarity": 0,
                "hist_similarity": best_first_pass["score"],
            }

        # Sort and get top histogram matches
        hist_scores.sort(key=lambda x: x["score"], reverse=True)
        top_matches = hist_scores[: config.num_top_matches]

        # Second pass: Calculate SSIM for top matches
        logger.debug(
            f"Found {len(top_matches)} histogram matches, running SSIM on top matches..."
        )
        resized_piece = PieceMatcher._resize_image(piece["img"])
        ssim_scores, best_second_pass = find_best_ssim_matches(
            resized_piece, top_matches, processed_set_pieces, config
        )

        # If no good matches found, try fallback matching
        if not ssim_scores:
            logger.debug(
                f"No matches found for {piece['piece']} in step {piece['step']} "
                f"on page {piece['page']}. Running SSIM on all set pieces."
            )
            ssim_scores, best_second_pass = fallback_ssim_matching(
                resized_piece, processed_set_pieces, config
            )

        if not ssim_scores:
            logger.debug(
                f"No matches found for {piece['piece']} in step {piece['step']} "
                f"on page {piece['page']} after fallback SSIM."
            )
        else:
            logger.debug(
                f"Found {len(ssim_scores)} SSIM matches above threshold {config.final_threshold}"
            )

        # Return best match
        best_match = (
            best_second_pass
            if not ssim_scores
            else max(ssim_scores, key=lambda x: x["final_similarity"])
        )

        logger.debug(
            f"Best match for piece {piece['piece']} in step {piece['step']} on page {piece['page']}: "
            f"Element ID: {best_match['element_id']}, SSIM: {best_match['final_similarity']:.3f}"
        )
        return {
            "piece": piece,
            "element_id": best_match["element_id"],
            "final_similarity": best_match["final_similarity"],
            "hist_similarity": best_match["hist_similarity"],
        }

    except Exception as e:
        logger.error(f"Error during two pass processing: {e}", exc_info=True)
        raise


def _remove_background(img, tolerance=20, debug=False):
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
        self.matched_step_pieces = {}  # Matching results
        self.unmatched_step_pieces = {}  # Unmatched results

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
        set_pieces_dir : Path or str
            Directory containing the set piece render images
        """

        for file_path in set_pieces_dir.glob("*"):
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
        Extract element ID from set_pieces filename.

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
        num_top_matches=10,
    ):
        """
        Match step pieces to set pieces using color histograms and SSIM.
        """
        logger.debug(
            f"Matching pieces with hist_threshold={hist_threshold}, "
            f"similarity_threshold={similarity_threshold}, n_workers={n_workers}"
        )
        # Reset results
        self.matched_step_pieces = {}
        self.unmatched_step_pieces = {}

        # Pre-process set pieces (compute histograms)
        processed_set_pieces = {}
        for element_id, pieces in self.set_pieces.items():
            processed_set_pieces[element_id] = []
            for piece in pieces:
                hist = self.extract_color_histogram(piece["image_data"])
                processed_set_pieces[element_id].append(
                    {
                        "image": self._resize_image(piece["image_data"]),
                        "histogram": hist,
                    }
                )

        # Process pieces in parallel
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
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
                        self.matched_step_pieces
                        if similarity > similarity_threshold
                        else self.unmatched_step_pieces
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
                    logger.error(f"Error processing piece: {e}")

        # Mark each set piece as matched or unmatched based on the results
        for element_id, pieces in self.set_pieces.items():
            if element_id not in self.matched_step_pieces:
                # If the element ID is not in matched_step_pieces, mark all as unmatched
                for piece in pieces:
                    piece["matched"] = False
            else:
                # If the element ID is in matched_step_pieces, mark as matched
                for piece in pieces:
                    piece["matched"] = True

        return self.matched_step_pieces, self.unmatched_step_pieces

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
        d = (
            self.unmatched_step_pieces
            if compare_unmatched
            else self.matched_step_pieces
        )
        if not d:
            logger.info("No results found. Run match_pieces() first.")
            return pd.DataFrame()

        # Create lists to store data for DataFrame
        data_rows = []

        for element_id, pages in d.items():
            # Find the corresponding set image
            if element_id not in self.set_pieces:
                continue

            set_piece_data = self.set_pieces[element_id][
                0
            ]  # Use the first render image
            set_piece_img = set_piece_data["image_data"]

            for page_num, steps_data in pages.items():
                for step_info in steps_data:
                    step_num = step_info["step"]
                    similarity = step_info["similarity"]
                    histogram_similarity = step_info["hist_similarity"]
                    piece_num = step_info["piece"]

                    # Find the corresponding piece image
                    step_piece_data = None
                    for piece in self.step_pieces:
                        if (
                            piece["page"] == page_num
                            and piece["step"] == step_num
                            and piece["piece"] == piece_num
                        ):
                            step_piece_data = piece
                            break

                    if step_piece_data is None:
                        continue

                    step_piece_img = step_piece_data["img"]

                    # Create HTML for images
                    step_piece_html = self._resize_image_for_display(
                        step_piece_img
                    )
                    set_piece_html = self._resize_image_for_display(
                        set_piece_img
                    )

                    # Extract and visualize color histograms
                    step_piece_hist = PieceMatcher.extract_color_histogram(
                        step_piece_img
                    )
                    set_piece_hist = PieceMatcher.extract_color_histogram(
                        set_piece_img
                    )

                    step_piece_hist_html = (
                        PieceMatcher.create_histogram_visualization(
                            step_piece_hist
                        )
                    )
                    set_piece_hist_html = (
                        PieceMatcher.create_histogram_visualization(
                            set_piece_hist
                        )
                    )

                    # Add row to data
                    data_rows.append(
                        {
                            "Element ID": element_id,
                            "Page": page_num,
                            "Step": step_num,
                            "Step Piece Image": step_piece_html,
                            "Set Piece Image": set_piece_html,
                            "Similarity Score": f"{similarity:.3f}",
                            "Step Piece Histogram": step_piece_hist_html,
                            "Set Piece Histogram": set_piece_hist_html,
                            "Histogram Similarity": f"{histogram_similarity:.3f}",
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
        if "Set Image" in clean_df.columns:
            clean_df["Set Image"] = "[IMAGE]"

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
        logger.info("Resizing image...")
        try:
            # Convert to PIL Image
            try:
                image_pil = Image.fromarray(img)
            except Exception:
                # If the image is not in a format that can be converted directly, use BytesIO
                image_pil = Image.open(BytesIO(img))

            # Resize the image
            resized_image = image_pil.resize(
                target_size, Image.Resampling.NEAREST
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
        processed_img = _remove_background(img, debug=False)

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

    @staticmethod
    def create_histogram_visualization(hist, bins=32):
        """
        Create a visualization of a color histogram.

        Parameters:
        ----
        hist : numpy.ndarray
            Flattened histogram array
        bins : int
            Number of bins per color channel

        Returns:
        ----
        str
            Base64-encoded image of the histogram visualization
        """
        # Reshape the flattened histogram back to 3D
        hist_3d = hist.reshape(bins, bins, bins)

        # Create RGB projections
        r_proj = np.sum(hist_3d, axis=(1, 2))
        g_proj = np.sum(hist_3d, axis=(0, 2))
        b_proj = np.sum(hist_3d, axis=(0, 1))

        # Normalize for visualization
        r_proj = r_proj / np.max(r_proj) if np.max(r_proj) > 0 else r_proj
        g_proj = g_proj / np.max(g_proj) if np.max(g_proj) > 0 else g_proj
        b_proj = b_proj / np.max(b_proj) if np.max(b_proj) > 0 else b_proj

        # Create the plot
        fig, ax = plt.subplots(figsize=(4, 2))
        x = np.arange(bins)

        ax.bar(x, r_proj, color="r", alpha=0.5, width=1.0)
        ax.bar(x, g_proj, color="g", alpha=0.5, width=1.0)
        ax.bar(x, b_proj, color="b", alpha=0.5, width=1.0)

        ax.set_xlim(0, bins)
        ax.set_ylim(0, 1.1)
        ax.set_title("RGB Color Distribution")
        ax.set_xticks([])
        ax.set_yticks([])

        # Convert plot to base64 image
        buf = BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)

        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        return f'<img src="data:image/png;base64,{img_str}"/>'

    def display_set_pieces_summary(self):
        """
        Display a comprehensive summary of all set pieces, including:
        - Element ID
        - Set piece image
        - Color histogram
        - Match status
        - Best match similarity (if matched)

        Returns:
        ----
        pd.DataFrame
            DataFrame containing the summary information
        """
        data_rows = []

        for element_id, pieces in self.set_pieces.items():
            # Get the first render image for this element
            set_piece_data = pieces[0]
            set_piece_img = set_piece_data["image_data"]

            # Create HTML for the set piece image
            set_piece_html = self._resize_image_for_display(set_piece_img)

            # Extract and visualize color histogram
            set_piece_hist = self.extract_color_histogram(set_piece_img)
            set_piece_hist_html = self.create_histogram_visualization(
                set_piece_hist
            )

            # Check if this element was matched in any step
            best_similarity = 0.0
            match_status = "Unmatched"

            if element_id in self.matched_step_pieces:
                match_status = "Matched"
                # Find the best similarity score for this element
                for page_matches in self.matched_step_pieces[
                    element_id
                ].values():
                    for match in page_matches:
                        best_similarity = max(
                            best_similarity, float(match["similarity"])
                        )

            # Add row to data
            data_rows.append(
                {
                    "Element ID": element_id,
                    "Set Piece Image": set_piece_html,
                    "Color Histogram": set_piece_hist_html,
                    "Match Status": match_status,
                    "Best Similarity": (
                        f"{best_similarity:.3f}"
                        if best_similarity > 0
                        else "N/A"
                    ),
                }
            )

        # Create DataFrame
        df = pd.DataFrame(data_rows)

        # Sort by match status (Matched first) and then by similarity score
        df = df.sort_values(
            by=["Match Status", "Best Similarity"], ascending=[True, False]
        )

        # Display the DataFrame with HTML rendering enabled
        display(HTML(df.to_html(escape=False)))

        # Print summary statistics
        total_pieces = len(df)
        matched_pieces = len(df[df["Match Status"] == "Matched"])
        unmatched_pieces = total_pieces - matched_pieces

        print(f"\nSummary Statistics:")
        print(f"Total Set Pieces: {total_pieces}")
        print(
            f"Matched Pieces: {matched_pieces} ({matched_pieces/total_pieces*100:.1f}%)"
        )
        print(
            f"Unmatched Pieces: {unmatched_pieces} ({unmatched_pieces/total_pieces*100:.1f}%)"
        )

        return df


def main():
    logging.basicConfig(level=logging.INFO)

    # Code profiling
    from cProfile import Profile
    from pstats import SortKey, Stats
    import time

    st = time.time()
    with Profile() as profile:
        manual = "6285327"
        """Example usage of the PieceMatcher class."""
        step_pieces_dir = Path(f"data/processed_booklets/{manual}/step pieces")
        set_pieces_dir = Path(
            f"data/processed_booklets/{manual}/identified set pieces"
        )

        # Initialize the matcher
        matcher = PieceMatcher()

        # Load step pieces and set pieces from directories
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

    et = time.time()
    elapsed_time = et - st
    print("Execution time:", elapsed_time, "seconds")


if __name__ == "__main__":
    main()
