import cv2
from pathlib import Path
import re
import pandas as pd
from io import BytesIO
from PIL import Image
import numpy as np
from IPython.display import display, HTML
import base64
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Dict, List, TypedDict, Union

logger = logging.getLogger(__name__)


class MatchResult(TypedDict):
    """Type definition for final matching results"""

    piece: dict
    element_id: str
    similarity: float


class StreamPool:
    """A pool of CUDA streams for template matching."""

    def __init__(self, n_streams=4):
        """Initialize the stream pool.

        Args:
            n_streams: Number of streams to maintain in the pool
        """
        self.streams = [cv2.cuda.Stream() for _ in range(n_streams)]
        self.current = 0

    def __del__(self):
        self.streams = None  # Streams are automatically destroyed

    def get_stream(self):
        """Get the next available stream in round-robin fashion."""
        stream = self.streams[self.current]
        self.current = (self.current + 1) % len(self.streams)
        return stream

    def wait_all(self):
        """Wait for all streams to complete."""
        for stream in self.streams:
            stream.waitForCompletion()


def calculate_template_matching_score_gpu(
    gpu_step_piece: cv2.cuda_GpuMat,
    set_piece_gpu_mat: cv2.cuda_GpuMat,
    stream: cv2.cuda.Stream = None,
) -> float:
    """Calculate template matching score using GPU."""
    try:
        # Verify GPU matrices
        logger.debug(f"GPU step piece type: {gpu_step_piece.type()}")
        logger.debug(f"GPU set piece type: {set_piece_gpu_mat.type()}")
        logger.debug(f"GPU step piece size: {gpu_step_piece.size()}")
        logger.debug(f"GPU set piece size: {set_piece_gpu_mat.size()}")
        # Create matcher
        matcher = cv2.cuda.createTemplateMatching(cv2.CV_32F, cv2.TM_SQDIFF)

        # Perform matching with error checking
        try:
            gpu_result = matcher.match(
                gpu_step_piece, set_piece_gpu_mat, stream=stream
            )
            result = gpu_result.download()
            score = float(result.min())
        except cv2.error as e:
            logger.error(f"Template matching failed: {e}")
            raise
        finally:
            # Clean up GPU resources
            if "gpu_result" in locals():
                gpu_result.release()

        return score

    except Exception as e:
        logger.error(f"GPU template matching error: {e}")
        raise


def calculate_template_matching_score(
    step_piece_img: np.ndarray, set_piece_img: np.ndarray
) -> float:
    """Calculate template matching score between two images"""
    result = cv2.matchTemplate(
        step_piece_img, set_piece_img, cv2.TM_SQDIFF_NORMED
    )
    min_val, max_val, _, _ = cv2.minMaxLoc(result)
    return min_val


def template_match_candidates(
    resized_step_piece: Union[np.ndarray, cv2.cuda_GpuMat],
    candidates: List[dict],
    resized_set_pieces: Dict,
    cuda_available: bool,
    stream: cv2.cuda.Stream = None,
) -> dict:
    """
    Calculate template matching for a list of candidate matches.
    """
    logger.debug("Calculating template matching for candidates...")

    # Initialize lists to store results
    best_match = None
    best_template_score = float("inf")
    # best_template_score = 0.0

    for candidate in candidates:
        element_id = candidate["element_id"]

        if cuda_available:
            # Use GPU matrices directly
            template_score = calculate_template_matching_score_gpu(
                resized_step_piece, candidate["gpu_mat"], stream
            )
        else:
            set_piece_idx = candidate.get("set_piece_idx", 0)
            set_piece_data = resized_set_pieces[element_id][set_piece_idx]
            # Calculate template matching scores using CPU arrays
            template_score = calculate_template_matching_score(
                resized_step_piece, set_piece_data
            )

        match_result = {
            "element_id": element_id,
            "similarity": template_score,
        }

        # Update best match based on the score
        if template_score < best_template_score:
            best_template_score = template_score
            best_match = match_result

    return best_match


def perform_template_matching(
    resized_step_piece: Union[np.ndarray, cv2.cuda.GpuMat],
    resized_set_pieces: Dict,
    cuda_available: bool,
    stream: cv2.cuda.Stream = None,
) -> dict:
    """
    Run template matching on all set pieces for a given step piece.
    """
    if cuda_available:
        # For GPU mode, resized_step_piece is a GpuMat
        all_candidates = [
            {
                "element_id": element_id,
                "set_piece_idx": idx,
                "gpu_mat": gpu_mat,
            }
            for element_id, pieces in resized_set_pieces.items()
            for idx, gpu_mat in enumerate(pieces)
        ]
    else:
        all_candidates = [
            {"element_id": element_id, "set_piece_idx": idx}
            for element_id, pieces in resized_set_pieces.items()
            for idx in range(len(pieces))
        ]
    return template_match_candidates(
        resized_step_piece,
        all_candidates,
        resized_set_pieces,
        cuda_available,
        stream,
    )


def _process_piece(args: tuple) -> MatchResult:
    """
    Process a single step piece using template matching.
    """
    (piece, resized_set_pieces, cuda_available, stream) = args

    try:
        if cuda_available:
            # For GPU mode, piece is a GpuMat, but we need the metadata
            # Assuming piece is a dictionary with 'gpu_mat' and metadata
            step_piece_gpu = piece["gpu_mat"]

            best_match = perform_template_matching(
                step_piece_gpu, resized_set_pieces, cuda_available, stream
            )

            metadata = {
                "page": piece["page"],
                "step": piece["step"],
                "piece": piece["piece"],
            }

            return {
                "piece": metadata,
                "element_id": best_match["element_id"],
                "similarity": best_match["similarity"],
            }
        else:
            # CPU mode - existing code
            resized_step_piece = PieceMatcher._resize_image(
                piece["img"], use_gpu=cuda_available
            )

            best_match = perform_template_matching(
                resized_step_piece, resized_set_pieces, cuda_available
            )

            return {
                "piece": piece,
                "element_id": best_match["element_id"],
                "similarity": best_match["similarity"],
            }

    except Exception as e:
        logger.error(f"Error during piece processing: {e}", exc_info=True)
        raise


def convert_threshold(threshold: float, image: cv2.cuda_GpuMat):
    """
    Convert threshold from relative to a nominal value
    """
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0")
    max_pixel_diff = 255.0
    max_sqdiff_score = (
        image.size()[0] * image.size()[1] * max_pixel_diff * max_pixel_diff
    )
    # Calculate the threshold value based on the image size
    return threshold * max_sqdiff_score


class PieceMatcher:
    """Class for matching LEGO pieces from instruction steps to set piece
    renders."""

    def __init__(self):
        """Initialize the PieceMatcher class."""
        # Check CUDA availability
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        # self.cuda_available = False
        try:
            # Print OpenCV version and CUDA information
            logger.info(f"OpenCV version: {cv2.__version__}")
            logger.info(
                f"CUDA device count: {cv2.cuda.getCudaEnabledDeviceCount()}"
            )

            # Set device before any other CUDA operations
            cv2.cuda.setDevice(0)

            # Warm up CUDA context
            # cv2.cuda.resetDevice()

        except Exception as e:
            logger.warning(f"CUDA initialization failed: {e}")
        # Store pieces and renders in memory
        # Format: [{"img": img_array, "page": page_num, "step": step_num, "piece": piece_num}]
        self.step_pieces = []
        self.set_pieces = (
            {}
        )  # Format: {element_id: [{"image_data": img_array}]}
        self.matched_step_pieces = {}  # Matching results
        self.unmatched_step_pieces = {}  # Unmatched results

        # GPU matrices for processed images
        self.set_pieces_gpu = {}  # {element_id: [GpuMat, ...]}
        self.step_pieces_gpu = []  # List of GpuMat objects

    def __del__(self):
        """Cleanup GPU resources"""
        self._clear_gpu_matrices()

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

                self.set_pieces[element_id].append({"image_data": img})

        logger.info(
            f"Loaded {len(self.set_pieces)} set pieces from {set_pieces_dir}"
        )

    def add_set_pieces(self, set_pieces):
        """
        Add set piece images to the matcher.

        Parameters:
        -----------
        set_pieces : dict
            Dictionary with element IDs as keys and lists of image data as
            values
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

    def match_pieces(self, threshold=0.15, n_workers=16):
        """
        Match step pieces to set pieces using template matching
        Uses stream pools for CUDA and multiple threads for CPU.

        Parameters:
        ----
        threshold : float, optional
            Similarity threshold for matching
        n_workers : int, optional
            Number of worker threads to use for CPU processing
        """
        logger.debug(
            f"Matching pieces with parameters: "
            f"threshold={threshold}, n_workers={n_workers}"
        )
        # Reset results
        self.matched_step_pieces = {}
        self.unmatched_step_pieces = {}

        # Resize set piece images for template matching
        resized_set_pieces = {
            element_id: list(
                map(lambda x: self._resize_image(x["image_data"]), pieces)
            )
            for element_id, pieces in self.set_pieces.items()
        }

        # Prepare GPU matrices if CUDA is available
        self._prepare_gpu_matrices(resized_set_pieces)

        try:
            step_pieces = (
                self.step_pieces_gpu
                if self.cuda_available
                else self.step_pieces
            )
            set_pieces = (
                self.set_pieces_gpu
                if self.cuda_available
                else resized_set_pieces
            )

            if self.cuda_available:
                results = []
                # stream pools processing for CUDA
                logger.info("Using CUDA with stream pools")
                # Create stream pool
                stream_pool = StreamPool(n_streams=100)
                threshold = convert_threshold(
                    threshold, step_pieces[0]["gpu_mat"]
                )

                for piece in step_pieces:
                    stream = stream_pool.get_stream()
                    args = (piece, set_pieces, self.cuda_available, stream)
                    results.append(_process_piece(args))

                # Wait for all streams to complete
                stream_pool.wait_all()
                # Process results
                for result in results:
                    self._process_result(result, threshold)
            else:
                # Multi-threaded processing for CPU
                logger.info(
                    f"Using multi-threaded processing with {n_workers} workers"
                )
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = []
                    for piece in step_pieces:
                        args = (piece, set_pieces, self.cuda_available, None)
                        futures.append(executor.submit(_process_piece, args))

                    # Collect results
                    for future in futures:
                        try:
                            result = future.result()
                            self._process_result(result, threshold)
                        except Exception as e:
                            logger.error(f"Error processing piece: {e}")
        except Exception as e:
            logger.error(f"Error during matching: {e}")

        finally:
            # Clean up GPU resources
            self._clear_gpu_matrices()

        # Mark each set piece as matched or unmatched based on the results
        for element_id, pieces in self.set_pieces.items():
            if element_id not in self.matched_step_pieces:
                # If the element ID is not in matched_step_pieces, mark all as
                # unmatched
                for piece in pieces:
                    piece["matched"] = False
            else:
                # If the element ID is in matched_step_pieces, mark as matched
                for piece in pieces:
                    piece["matched"] = True

        return self.matched_step_pieces, self.unmatched_step_pieces

    def _process_result(self, result, threshold):
        """
        Process a single result from piece matching.

        Parameters:
        ----
        result : dict
            Result from _process_piece
        threshold : float
            Similarity threshold for matching
        """
        piece = result["piece"]
        element_id = result["element_id"]
        similarity = result["similarity"]
        page_num = piece["page"]
        step_num = piece["step"]
        piece_num = piece["piece"]

        # Choose the appropriate dictionary based on the similarity score
        d = (
            self.matched_step_pieces
            if similarity < threshold
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
            }
        )

    def _resize_image_for_display(self, img, max_size=(100, 100)):
        """Resize image and convert to base64 for HTML display"""
        try:
            img_pil = Image.fromarray(img)
        except Exception:
            # If the image is not in a format that can be converted directly,
            # use BytesIO
            img_pil = Image.open(BytesIO(img))
        img_pil.thumbnail(max_size)
        img_bio = BytesIO()
        img_pil.save(img_bio, format="PNG")
        img_b64 = base64.b64encode(img_bio.getvalue()).decode("utf-8")
        return f'<img src="data:image/png;base64,{img_b64}"/>'

    def create_comparison_dataframe(self, compare_unmatched=False):
        """
        Create a Pandas DataFrame showing matched/unmatched pieces and their
        similarity scores using data from the matched/unmatched dictionaries
        and the stored images.

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

                    # Add row to data
                    data_rows.append(
                        {
                            "Element ID": element_id,
                            "Page": page_num,
                            "Step": step_num,
                            "Step Piece Image": step_piece_html,
                            "Set Piece Image": set_piece_html,
                            "Similarity Score": f"{step_info['similarity']:.3f}",
                        }
                    )

        # Create DataFrame
        df = pd.DataFrame(data_rows)

        return df

    def display_comparison_dataframe(self, df=None, compare_unmatched=False):
        """
        Display the comparison DataFrame with properly rendered images in a
        Jupyter notebook

        Parameters:
        -----------
        df : pd.DataFrame, optional
            DataFrame created by create_comparison_dataframe. If None, creates a
            new one.

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

    def display_set_pieces_summary(self):
        """
        Display a comprehensive summary of all set pieces, including:
        - Element ID
        - Set piece image
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
                            best_similarity,
                            float(match["similarity"]),
                        )

            # Add row to data
            data_rows.append(
                {
                    "Element ID": element_id,
                    "Set Piece Image": set_piece_html,
                    "Match Status": match_status,
                    "Best Template Similarity": (
                        f"{match['similarity']:.3f}"
                        if best_similarity > 0
                        else "N/A"
                    ),
                }
            )

        # Create DataFrame
        df = pd.DataFrame(data_rows)

        # Sort by match status (Matched first) and then by similarity score
        df = df.sort_values(
            by=["Match Status", "Best Template Similarity"],
            ascending=[True, False],
        )

        # Display the DataFrame with HTML rendering enabled
        display(HTML(df.to_html(escape=False)))

        # Print summary statistics
        total_pieces = len(df)
        matched_pieces = len(df[df["Match Status"] == "Matched"])
        unmatched_pieces = total_pieces - matched_pieces

        logger.info("Summary Statistics:")
        logger.info(f"Total Set Pieces: {total_pieces}")
        logger.info(
            f"Matched Pieces: {matched_pieces} ({matched_pieces/total_pieces*100:.1f}%)"
        )
        logger.info(
            f"Unmatched Pieces: {unmatched_pieces} ({unmatched_pieces/total_pieces*100:.1f}%)"
        )

        return df

    def _prepare_gpu_matrices(self, processed_set_pieces: Dict):
        """
        Prepare GPU matrices for all images before matching.
        Should be called at the start of match_pieces().
        """
        if not self.cuda_available:
            return

        logger.info("Preparing GPU matrices for matching...")
        try:
            # Clear any existing GPU matrices
            self._clear_gpu_matrices()

            # Upload set pieces to GPU
            for element_id, pieces in processed_set_pieces.items():
                self.set_pieces_gpu[element_id] = []
                for piece in pieces:
                    if not isinstance(piece, np.ndarray):
                        piece = np.array(piece)

                    # Convert to grayscale if needed
                    if len(piece.shape) == 3:
                        piece = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)

                    # Upload to GPU
                    gpu_mat = cv2.cuda_GpuMat()
                    gpu_mat.upload(piece.astype(np.float32))
                    self.set_pieces_gpu[element_id].append(gpu_mat)

            # Upload step pieces to GPU
            self.step_pieces_gpu = []
            for piece in self.step_pieces:
                img = piece["img"]

                # Convert to grayscale if needed
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Resize image
                resized_img = self._resize_image(img)

                # Upload to GPU
                gpu_mat = cv2.cuda_GpuMat()
                gpu_mat.upload(resized_img.astype(np.float32))

                self.step_pieces_gpu.append(
                    {
                        "gpu_mat": gpu_mat,
                        "page": piece["page"],
                        "step": piece["step"],
                        "piece": piece["piece"],
                    }
                )

            logger.info(
                f"Uploaded {sum(len(pieces) for pieces in self.set_pieces_gpu.values())} "
                f"set pieces and {len(self.step_pieces_gpu)} step pieces to GPU"
            )
        except Exception as e:
            logger.error(f"Error preparing GPU matrices: {e}")
            raise

    def _clear_gpu_matrices(self):
        """Release all GPU matrices"""
        if not self.cuda_available:
            return

        # Release set pieces
        for element_id in self.set_pieces_gpu:
            for gpu_mat in self.set_pieces_gpu[element_id]:
                gpu_mat.release()
        self.set_pieces_gpu.clear()

        # Release step pieces
        for gpu_mat in self.step_pieces_gpu:
            gpu_mat["gpu_mat"].release()
        self.step_pieces_gpu.clear()

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
            ]

        # Display the DataFrame
        # Create the summary DataFrame from the dictionary
        summary_df = pd.DataFrame(
            summary,
            index=[
                "Total Pieces",
                "Min Similarity Score",
                "Max Similarity Score",
                "Mean Similarity Score",
            ],
        )

        display(summary_df)
        return summary_df

    @staticmethod
    def _resize_image(img, target_size=(200, 200), use_gpu=False):
        """
        Resize an image to a standard size.

        Parameters:
        -----------
        img : Union[bytes, cv2.cuda_GpuMat]
            Input image, either on CPU or GPU
        target_size : tuple, optional
            Target size (width, height)

        Returns:
        --------
        Union[numpy.ndarray, cv2.cuda_GpuMat]
            Resized image in the same format as input (GPU or CPU)
        """
        logger.debug("Resizing image...")
        try:
            # Resize the image
            if use_gpu:
                resized_image = cv2.cuda.resize(
                    img, target_size, interpolation=cv2.INTER_NEAREST
                )
            else:
                # Convert to PIL Image
                try:
                    image_pil = Image.fromarray(img)
                except Exception:
                    # If the image is not in a format that can be converted
                    # directly, use BytesIO
                    image_pil = Image.open(BytesIO(img))
                # Resize using OpenCV on CPU

                resized_image = image_pil.resize(
                    target_size, Image.Resampling.NEAREST
                )

            # Convert to RGB if it's not already
            # if resized_image.mode != "RGB":
            #    resized_image = resized_image.convert("RGB")

            # return img_byte_arr.getvalue()
            return np.array(resized_image)

        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return None


def main():
    logging.basicConfig(level=logging.INFO)

    print(f"OpenCV version: {cv2.__version__}")

    from cProfile import Profile
    from pstats import SortKey, Stats
    import time

    st = time.time()
    with Profile() as profile:
        manual = "6519906"
        """Example usage of the PieceMatcher class."""
        processed_manual_dir = Path(f"data/processed manuals/{manual}")
        step_pieces_dir = processed_manual_dir / "step pieces"
        set_pieces_dir = processed_manual_dir / "identified set pieces"

        # Initialize the matcher
        matcher = PieceMatcher()

        # Load step pieces and set pieces from directories
        matcher.load_step_pieces_from_directory(step_pieces_dir)
        matcher.load_set_pieces_from_directory(set_pieces_dir)

        # Match pieces
        matched, unmatched = matcher.match_pieces()

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
