from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional
from lego_step_detector import LEGOStepDetector
from dataset import LEGOStepDataset
from utils.json_merger import merge_json_files

import logging
logger = logging.getLogger(__name__)

class LEGOModelTrainer:
    def __init__(
        self,
        images_dir: Path,
        annotations_dir: Path,
        model_save_path: Path,
        batch_size: int = 2,
        num_workers: int = 4
    ):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.model_save_path = Path(model_save_path)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.model = LEGOStepDetector()
        self.dataset: Optional[LEGOStepDataset] = None
        self.dataloader: Optional[DataLoader] = None

    def prepare_data(self):
        """Set up dataset and dataloader"""
        # Merge annotation files
        annotation_file_path = self.annotations_dir / "merged" / "annotations.json"
        merge_json_files(
            input_folder=self.annotations_dir,
            output_path=annotation_file_path
        )

        # Configure transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Create dataset
        self.dataset = LEGOStepDataset(
            image_dir=self.images_dir,
            annotation_file=annotation_file_path,
            transform=transform
        )

        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.model._collate_fn
        )

    def train(self, num_epochs: int = 10):
        """Run the training process"""
        if self.dataset is None:
            self.prepare_data()

        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Dataset size: {len(self.dataset)} images")

        self.model.train(self.dataset, num_epochs)
        self.model.save_model(self.model_save_path)
        logger.info("Training completed successfully")

def train_model_cli():
    """CLI entry point for model training"""
    import argparse

    parser = argparse.ArgumentParser(description='Train LEGO Step Detector')
    parser.add_argument('--images-dir', type=Path, required=True)
    parser.add_argument('--annotations-dir', type=Path, required=True)
    parser.add_argument('--model-save-path', type=Path, required=True)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=4)

    args = parser.parse_args()

    trainer = LEGOModelTrainer(
        images_dir=args.images_dir,
        annotations_dir=args.annotations_dir,
        model_save_path=args.model_save_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    trainer.train(num_epochs=args.num_epochs)

if __name__ == "__main__":
    from logger import setup_logging
    setup_logging(logging.DEBUG)
    train_model_cli()