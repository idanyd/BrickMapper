from pathlib import Path
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from typing import Dict, List

import logging
logger = logging.getLogger(__name__)

class LEGOStepDataset(Dataset):
    """Custom dataset for LEGO instruction steps."""

    def __init__(self, image_dir: Path, annotation_file: Path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.annotations = self._load_annotations(annotation_file)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load image and annotations
        image = self.annotations[idx]['filename']
        set_num = image.split('_')[0]
        booklet_num = image.split('_')[1]
        img_path = self.image_dir / set_num / booklet_num / image
        image = Image.open(img_path).convert('RGB')

        # Get bounding boxes and labels
        boxes = self.annotations[idx]['boxes']
        labels = self.annotations[idx]['labels']

        # Create target dict
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        if self.transform:
            image = self.transform(image)

        return image, target

    def _load_annotations(self, annotation_file: Path) -> List[Dict]:
        """Load and parse annotation file."""
        try:
            with annotation_file.open() as f:
                annotations = json.load(f)
        except FileNotFoundError:
            logger.error(f"Annotation file not found: {annotation_file}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in annotation file: {annotation_file}")
            return []

        processed_annotations = []
        for ann in annotations['data']:
            # Create empty tensors if no boxes/labels
            boxes = torch.tensor(ann['boxes'], dtype=torch.float32) if ann['boxes'] else torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.tensor(ann['labels'], dtype=torch.int64) if ann['labels'] else torch.zeros((0,), dtype=torch.int64)


            processed_annotations.append({
                'filename': ann['image_path'],
                'boxes': boxes,
                'labels': labels
            })

        return processed_annotations