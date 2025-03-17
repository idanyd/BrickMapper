import os
from dotenv import load_dotenv
import neptune
from ultralytics import YOLO
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from PIL import Image
from utils.numbers_ocr import extract_numbers
from enum import Enum

import logging

logger = logging.getLogger(__name__)

ANNOTATIONS_PATH = Path("data/training/annotations/data.yaml")
BEST_MODEL_PATH = Path("runs/train/exp/weights/best.pt")
ANNOTATED_IMGS_DIR = Path("runs/validate/test_evaluation/annotated_images")


# Annotation classes
class BoxClass(Enum):
    STEP_BOX = 0
    STEP_NUMBER = 1


class StepDetector:
    def __init__(self):
        self.model = (
            YOLO(BEST_MODEL_PATH) if BEST_MODEL_PATH.exists() else None
        )

    @staticmethod
    def init_run(tags=None):
        """
        Initialize a Neptune run for experiment tracking.

        Args:
            tags (list, optional): List of tags to associate with the run. Defaults to None.

        Returns:
            neptune.Run: Initialized Neptune run object.
        """

        # Load environment variables
        load_dotenv()
        run = neptune.init_run(
            project=os.getenv("NEPTUNE_PROJECT"),
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
            tags=tags,
        )
        return run

    @staticmethod
    def _find_matching_step_box(step_number_box, step_boxes):
        """
        find the index of the nearest box in step_boxes above step_number_box,
        where "nearest" is defined by the Chebyshev distance between the bottom-left corner
        of the upper box and the top-left corner of the current box.

        Args:
            step_number_box: Box containing the step number
            step_boxes: List of boxes, where each box is [x1,y1,x2,y2]

        Returns:
            The index of the nearest box above step_number_box.
            None if no box is above.
        """

        def chebyshev_distance(x1, y1, x2, y2):
            """Calculate Chebyshev distance between two points"""
            return max(abs(x1 - x2), abs(y1 - y2))

        sb_x1, sb_y1, _, _ = step_number_box.xyxy[0].cpu().numpy()
        min_distance = float("inf")
        nearest_idx = None

        for i, box in enumerate(step_boxes):

            cb_x1, cb_y2 = box[0], box[3]  # (x1, y2)

            # Check if candidate box is above current box
            if box[3] <= sb_y1:  # y2 of candidate <= y1 of current
                distance = chebyshev_distance(
                    sb_x1,
                    sb_y1,
                    cb_x1,
                    cb_y2,
                )

                if distance < min_distance:
                    min_distance = distance
                    nearest_idx = i

        return nearest_idx

    def setup_yolo_training(
        self,
        data_yaml_path,
        model_size="n",  # n (nano), s (small), m (medium), l (large), x (xlarge)
        epochs=100,
        batch_size=16,
        imgsz=640,
    ):
        """
        Setup and train a YOLO model from scratch with specified parameters.

        Args:
            data_yaml_path (str): Path to the YAML file containing dataset configuration.
            model_size (str, optional): Size of YOLO model ('n','s','m','l','x'). Defaults to "n".
            epochs (int, optional): Number of training epochs. Defaults to 100.
            batch_size (int, optional): Training batch size. Defaults to 16.
            imgsz (int, optional): Input image size. Defaults to 640.

        Returns:
            dict: Training results containing metrics and performance data.
        """

        # Create a new YOLO model
        model_name = f"yolo11{model_size}.yaml"
        self.model = YOLO(model_name)  # build a new model from scratch

        # Configure training parameters
        training_args = {
            "data": data_yaml_path,  # path to data.yaml
            "epochs": epochs,  # number of epochs
            "batch": batch_size,  # batch size
            "imgsz": imgsz,  # image size
            "patience": 50,  # early stopping patience
            "device": "0",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            "workers": 8,  # number of worker threads
            "project": "runs/train",  # save results to project/name
            "name": "exp",  # save results to project/name
            "pretrained": False,  # start fresh
            "optimizer": "auto",  # optimizer (SGD, Adam, etc.)
            "verbose": True,  # verbose output
            "seed": 42,  # random seed
            "deterministic": True,  # deterministic mode
            "single_cls": False,  # train multi-class data as single-class
            "rect": False,  # rectangular training
            "cos_lr": True,  # cosine LR scheduler
            "close_mosaic": 10,  # disable mosaic augmentation for final 10 epochs
            "resume": False,  # resume training from last checkpoint
        }

        # Initialize Neptune run
        run = StepDetector.init_run(["yolo-detection"])

        # Train the model
        try:
            results = self.model.train(**training_args)

            # Log model configuration
            run["model/task"] = "training"
            run["model/name"] = model_name

            logger.info("Training completed successfully!")
            return results
        except Exception as e:
            logger.error(f"An error occurred during training: {str(e)}")
            return None

    def save_annotated_image(
        self,
        image_path,
        output_path,
        boxes_to_plot,
        plot_ground_truth_boxes=False,
    ):
        """
        Visualize predictions and ground truth boxes on an image and save the result.

        Args:
            img: Loaded image object
            image_path (str): Path to original image (needed for ground truth)
            output_path (str): Path where to save the annotated image
            boxes_to_plot (list): List of dictionaries containing box information
            plot_ground_truth_boxes (bool): Whether to plot ground truth boxes
        """

        # Load image
        img = Image.open(image_path)

        plt.figure(figsize=(12, 8))
        plt.imshow(img)

        # Plot all predicted boxes
        for box in boxes_to_plot:
            x1, y1, x2, y2 = box["coords"]
            plt.gca().add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    color="red",
                    linewidth=2,
                    label=f"Pred: Class {box['cls']} ({box['conf']:.2f})",
                )
            )

        # Plot ground truth boxes if requested
        if plot_ground_truth_boxes:
            label_path = (
                image_path.replace("images", "labels")
                .replace(".jpg", ".txt")
                .replace(".png", ".txt")
            )
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    labels = f.read().splitlines()

                img_height, img_width = img.shape[:2]
                for label in labels:
                    cls, x_center, y_center, width, height = map(
                        float, label.split()
                    )
                    # Convert normalized coordinates to pixel coordinates
                    x1 = int((x_center - width / 2) * img_width)
                    y1 = int((y_center - height / 2) * img_height)
                    w = int(width * img_width)
                    h = int(height * img_height)

                    # Draw ground truth box (green)
                    plt.gca().add_patch(
                        plt.Rectangle(
                            (x1, y1),
                            w,
                            h,
                            fill=False,
                            color="green",
                            linewidth=2,
                            label=f"GT: Class {int(cls)}",
                        )
                    )

        # Add legend and title
        plt.legend()
        title = (
            "Predictions vs Ground Truth:"
            if plot_ground_truth_boxes
            else "Predictions"
        )
        plt.title(f"{title} {Path(image_path).name}")
        plt.axis("off")

        # Save annotated image
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        logger.info(f"Saved annotated image to: {output_path}")

    def detect_steps(self, image, conf_threshold=0.25):
        """
        Detect steps in an image and return the bounding box coordinates.

        Args:
            image (PIL image): Input image to detect steps
            conf_threshold (float, optional): Confidence threshold for predictions. Defaults to 0.25

        Returns:
            list: List of dictionaries containing steps information
        """
        try:
            # Get predictions
            pred_results = self.model.predict(
                image, conf=conf_threshold, save=False, verbose=False
            )

            # Process predictions and collect steps
            steps = {}  # Store step info for visualization

            for result in pred_results:
                step_boxes = [
                    box.xyxy[0].cpu().numpy()
                    for box in result.boxes
                    if box.cls == BoxClass.STEP_BOX.value
                ]

                if len(step_boxes) == 0:
                    logger.debug("No step boxes detected in the image")
                    continue

                step_numbers = [
                    box
                    for box in result.boxes
                    if box.cls == BoxClass.STEP_NUMBER.value
                ]

                for step_number_box in step_numbers:
                    # Extract number from step
                    step_img = image.crop(
                        step_number_box.xyxy[0].cpu().numpy()
                    )
                    step_number = extract_numbers(step_img)
                    if not step_number:
                        continue
                    step_box_index = self._find_matching_step_box(
                        step_number_box, step_boxes
                    )
                    if step_box_index is None:
                        logger.debug(
                            f"No matching step box found for step number: {step_number}"
                        )
                    else:
                        logger.info(f"Detected step number: {step_number}")

                        # Store box info for plotting
                        steps[step_number] = step_boxes[step_box_index]

            return steps

        except Exception as e:
            logger.error(f"An error occurred while annotating image: {str(e)}")
            return None

    def save_annotated_images(self, data_yaml_path, output_dir):
        """
        Save images with both predicted and ground truth bounding box annotations.

        Args:
            data_yaml_path (str): Path to the data configuration YAML file.
            output_dir (Path): Directory to save annotated images.
        """

        # Load data.yaml to get test file path
        with open(data_yaml_path, "r") as f:
            data_config = yaml.safe_load(f)

        # Get test file paths
        with open(data_config["test"], "r") as f:
            test_images = f.read().splitlines()

        # Process each test image
        for img_path in test_images:
            steps = self.detect_steps(img_path)

            self.save_annotated_image(
                img_path,
                output_dir / Path(img_path).name,
                steps,
                plot_ground_truth_boxes=True,
            )

    def test_model(
        self,
        data_yaml_path=ANNOTATIONS_PATH,
        conf_threshold=0.25,
        save_results=True,
    ):
        """
        Test the trained YOLO model using test data specified in data.yaml

        Args:
            model_path (str): Path to the trained model weights
            data_yaml_path (Path): Path to data.yaml file
            conf_threshold (float): Confidence threshold for predictions
            save_results (bool): Whether to save prediction results

        Returns:
            dict: Dictionary containing test results
        """
        try:
            # Load data.yaml configuration
            with open(data_yaml_path, "r") as f:
                data_config = yaml.safe_load(f)

            # Get test data path from data.yaml
            test_path = data_config.get("test")
            if not test_path:
                raise ValueError("Test path not found in data.yaml")

            # Run inference on test data
            results = self.model.predict(
                source=test_path,
                conf=conf_threshold,
                save=save_results,
                project="runs/predict",
                name="test_results",
            )

            # Process and log results
            logger.info("Test Results:")
            for r in results:
                logger.info(f"Image: {Path(r.path).name}")
                logger.info(f"Detections: {len(r.boxes)} objects found")

                # log detection details
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    logger.info(f"Class: {cls}, Confidence: {conf:.2f}")

            return results

        except Exception as e:
            logger.error(f"An error occurred during testing: {str(e)}")
            return None

    def evaluate_model(
        self,
        data_yaml_path=ANNOTATIONS_PATH,
        save_results=True,
        plot_confusion_matrix=True,
        should_save_annotated_images=True,
    ):
        """
        Evaluate the trained YOLO model and save annotated images

        Args:
            model_path (str): Path to the trained model weights
            data_yaml_path (Path): Path to data.yaml file
            save_results (bool): Whether to save validation results
            plot_confusion_matrix (bool): Whether to generate confusion matrix
            save_annotated_images (bool): Whether to save images with annotations

        """
        try:
            # Create output directory for annotated images
            ANNOTATED_IMGS_DIR.mkdir(parents=True, exist_ok=True)

            # Run validation on test dataset
            results = self.model.val(
                data=data_yaml_path,
                split="test",  # Use test split from data.yaml
                save_json=save_results,
                save_hybrid=save_results,
                plots=True,
                project="runs/validate",
                name="test_evaluation",
            )

            if should_save_annotated_images:
                StepDetector.save_annotated_images(
                    data_yaml_path, ANNOTATED_IMGS_DIR
                )

            # Log detailed metrics
            logger.info(f"mAP50: {results.box.map50:.4f}")  # mAP at IoU 0.50
            logger.info(
                f"mAP50-95: {results.box.map:.4f}"
            )  # mAP at IoU 0.50:0.95
            logger.info(f"Precision: {results.box.p:.4f}")  # Precision
            logger.info(f"Recall: {results.box.r:.4f}")  # Recall
            logger.info(f"F1-Score: {results.box.f1:.4f}")  # F1 Score

            # Log per-class metrics
            logger.info("Per-class Performance:")
            for i, ap50 in enumerate(results.box.ap50):
                logger.info(f"Class {i} - AP50: {ap50:.4f}")

            # Initialize Neptune run for logging if needed
            run = StepDetector.init_run(["yolo-evaluation"])

            # Log metrics to Neptune
            run["metrics/mAP50"] = results.box.map50
            run["metrics/mAP50-95"] = results.box.map
            run["metrics/precision"] = results.box.p
            run["metrics/recall"] = results.box.r
            run["metrics/f1"] = results.box.f1

            # Log confusion matrix plot if generated
            if plot_confusion_matrix and os.path.exists(
                "runs/validate/test_evaluation/confusion_matrix.png"
            ):
                run["visualizations/confusion_matrix"].upload(
                    "runs/validate/test_evaluation/confusion_matrix.png"
                )

            return results

        except Exception as e:
            logger.error(f"An error occurred during evaluation: {str(e)}")
            return None


def main(train=False, evaluate=False, test=False):
    """
    Main execution function to control the training, evaluation, and testing pipeline.

    Args:
        train (bool, optional): Whether to train the model. Defaults to False.
        evaluate (bool, optional): Whether to evaluate the model. Defaults to False.
        test (bool, optional): Whether to test the model. Defaults to False.

    Raises:
        FileNotFoundError: If data.yaml file is not found at specified path.
    """
    current_dir = Path().resolve()

    print(current_dir)
    # Verify data.yaml exists
    if not os.path.exists(ANNOTATIONS_PATH):
        raise FileNotFoundError(f"data.yaml not found at {ANNOTATIONS_PATH}")

    detector = StepDetector()

    if train:
        # Start training
        detector.setup_yolo_training(data_yaml_path=ANNOTATIONS_PATH)

    if evaluate:
        # Evaluate the model
        detector.evaluate_model(
            data_yaml_path=ANNOTATIONS_PATH, should_save_annotated_images=False
        )

    if test:
        try:
            # Test the trained model
            detector.test_model(data_yaml_path=ANNOTATIONS_PATH)
        except Exception as e:
            logger.error(f"An error occurred during testing: {str(e)}")


if __name__ == "__main__":
    main(evaluate=True)
