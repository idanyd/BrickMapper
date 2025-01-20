import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from utils.json_merger import merge_json_files
from dataset import LEGOStepDataset

logger = logging.getLogger(__name__)

ANNOTATIONS_FILE = "annotations.json"

class LEGOStepDetector:
    """ML model for detecting LEGO instruction steps."""
    
    def __init__(self, model_path: Optional[Path] = None):        
        # Initialize model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        logger.debug("Loaded pre-trained Faster R-CNN model")

        # Modify the box predictor for our classes
        # Class 1: Step Box, Class 2: Step Number
        num_classes = 3  # Background + 2 classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        logger.debug(f"Modified box predictor for {num_classes} classes")

        if model_path:
            self.load_model(model_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"Model initialized on device: {self.device}")
    
    def train(self, train_dataset: LEGOStepDataset, num_epochs: int = 10):
        """Train the model on LEGO instruction data."""
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Dataset size: {len(train_dataset)} images")

        # Set up data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # Set up optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005
        )
        logger.debug("Initialized optimizer")

        # Set the model to training mode
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0
            batch_count = 0

            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            for images, targets in train_loader:
                # Move images and targets to GPU if available
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
                batch_count += 1

                if batch_count % 10 == 0:  # Log every 10 batches
                    logger.debug(f"Epoch {epoch + 1}, Batch {batch_count}: Loss = {losses.item():.4f}")

            logger.info(f"Epoch {epoch + 1}: Loss = {epoch_loss / len(train_loader)}")
    
    def detect_steps(self, image: Image.Image) -> List[Dict]:
        """
        Detect step boxes and numbers in an image.
        
        Args:
            image: PIL Image of the page
            
        Returns:
            List of dicts containing detected steps and their numbers
        """
        self.model.eval()
        
        # Transform image for model
        transform = transforms.ToTensor()
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Process predictions
        steps_info = []
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # Group boxes by step
        steps = {}
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5:  # Confidence threshold
                if label == 1:  # Step box
                    steps[len(steps)] = {'bbox': box}
                elif label == 2:  # Step number
                    # Find closest step box
                    closest_step = self._find_closest_step(box, steps)
                    if closest_step is not None:
                        steps[closest_step]['number_bbox'] = box
        
        # Process each step
        for step_info in steps.values():
            if 'bbox' in step_info:
                step_dict = {
                    'bbox': tuple(step_info['bbox']),
                    'step_number': None
                }
                
                if 'number_bbox' in step_info:
                    # Extract and process number using OCR
                    number_region = self._extract_number_region(
                        image, step_info['number_bbox']
                    )
                    if number_region:
                        step_dict['step_number'] = self._process_number(number_region)
                
                steps_info.append(step_dict)
        
        return steps_info
    
    def save_model(self, path: Path):
        """
        Save model weights.

        Args:
            path: Path where the model weights will be saved
        """
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save the model
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path):
        """Load model weights."""
        try:
            self.model.load_state_dict(torch.load(path))
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {str(e)}")
            raise
    
    @staticmethod
    def _collate_fn(batch):
        """Custom collate function for data loader."""
        return tuple(zip(*batch))
    
    @staticmethod
    def _find_closest_step(number_box, steps):
        """Find the step box closest to a number box."""
        min_dist = float('inf')
        closest_step = None
        
        number_center = np.array([
            (number_box[0] + number_box[2]) / 2,
            (number_box[1] + number_box[3]) / 2
        ])
    
        for step_id, step_info in steps.items():
            if 'bbox' not in step_info:
                continue

            box = step_info['bbox']
            box_center = np.array([
                (box[0] + box[2]) / 2,
                (box[1] + box[3]) / 2
            ])
            
            dist = np.linalg.norm(box_center - number_center)
            if dist < min_dist:
                min_dist = dist
                closest_step = step_id
        
        return closest_step

    def _extract_number_region(self, image: Image.Image, bbox: np.ndarray) -> Optional[Image.Image]:
        """Extract the region containing a potential step number."""
        try:
            return image.crop(tuple(bbox))
        except Exception as e:
            logger.error(f"Error extracting number region: {str(e)}")
            return None

    def _process_number(self, number_image: Image.Image) -> Optional[int]:
        """Process a number image to extract the step number."""
        try:
            # Use OCR or number recognition model
            # TODO: Implement number recognition
            return None
        except Exception as e:
            logger.error(f"Error processing number: {str(e)}")
            return None

def train_model(
    images_dir: Path,
    model_save_path: Path,
    num_epochs: int = 10
):
    """Train the LEGO step detector model."""
    logger.info("Starting model training process")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Model save path: {model_save_path}")

    # Set up annotations directories
    annotations_dir = Path("data/training/annotations")
    annotation_file_path = annotations_dir / "merged" / ANNOTATIONS_FILE
    merge_json_files(
        input_folder=annotations_dir, output_path=annotation_file_path
    )
    logger.info(f"Merged annotations saved to {annotation_file_path}")

    # Data transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    logger.debug("Data transforms configured")

    # Create dataset
    dataset = LEGOStepDataset(
        image_dir=images_dir,
        annotation_file=annotation_file_path,
        transform=transform,
    )
    logger.info(f"Dataset created with {len(dataset)} images")

    # Initialize model
    detector = LEGOStepDetector()

    # Train model
    logger.info("Starting training...")
    detector.train(train_dataset=dataset, num_epochs=num_epochs)

    # Save trained model
    logger.info(f"Saving model to {model_save_path}")
    detector.save_model(model_save_path)