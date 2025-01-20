import pytest
import torch
import json
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch, MagicMock, mock_open
from src.dataset import LEGOStepDataset

@pytest.fixture
def mock_annotation_data():
    return {
        "data": [
            {
                "image_path": "12345_1_page1.jpg",
                "boxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
                "labels": [1, 2]
            },
            {
                "image_path": "12345_1_page2.jpg",
                "boxes": [],
                "labels": []
            }
        ]
    }

@pytest.fixture
def mock_filesystem():
    with patch("pathlib.Path") as mock_path_class:
        # Configure the Path class mock to return appropriate path objects
        def path_factory(path_str):
            mock_path = MagicMock(spec=Path)
            mock_path.__str__.return_value = str(path_str)
            # Make / operator return a new mock path with combined string
            mock_path.__truediv__.side_effect = lambda x: path_factory(f"{str(path_str)}/{x}")
            return mock_path

        # Make Path() calls return our configured mock paths
        mock_path_class.side_effect = path_factory

        # Create base paths
        mock_image_dir = path_factory("/path/to/images")
        mock_annotation_file = path_factory("/path/to/annotations.json")

        with patch("builtins.open", mock_open()) as mock_file:
            yield {
                "image_dir": mock_image_dir,
                "annotation_file": mock_annotation_file,
                "file": mock_file,
                "path_class": mock_path_class,  # Now being used properly
            }

@pytest.fixture
def mock_transform():
    return Mock(return_value=torch.rand(3, 100, 100))


class TestLEGOStepDataset:
    def test_initialization(self, mock_filesystem, mock_annotation_data):
        """Test dataset initialization"""
        with patch('json.load', return_value=mock_annotation_data):
            dataset = LEGOStepDataset(
                mock_filesystem["image_dir"],
                mock_filesystem["annotation_file"]
            )

            assert len(dataset.annotations) == 2
            assert isinstance(dataset.annotations, list)
            assert dataset.transform is None

            # Verify the loaded annotations match the expected structure
            assert isinstance(dataset.annotations[0]['boxes'], torch.Tensor)
            assert isinstance(dataset.annotations[0]['labels'], torch.Tensor)
            assert dataset.annotations[0]['filename'] == mock_annotation_data['data'][0]['image_path']

    def test_initialization_with_transform(self, mock_filesystem, mock_annotation_data, mock_transform):
        """Test dataset initialization with transform"""
        with patch('json.load', return_value=mock_annotation_data):
            dataset = LEGOStepDataset(
                mock_filesystem["image_dir"],
                mock_filesystem["annotation_file"],
                transform=mock_transform
            )

            assert dataset.transform == mock_transform

    def test_len(self, mock_filesystem, mock_annotation_data):
        """Test dataset length"""
        with patch('json.load', return_value=mock_annotation_data):
            dataset = LEGOStepDataset(
                mock_filesystem["image_dir"],
                mock_filesystem["annotation_file"]
            )

            assert len(dataset) == 2

    @patch('PIL.Image.open')
    def test_getitem(self, mock_image_open, mock_filesystem, mock_annotation_data):
        """Test getting an item from dataset"""
        with patch('json.load', return_value=mock_annotation_data):
            # Setup mock image
            mock_image = Mock(spec=Image.Image)
            mock_image.convert.return_value = mock_image
            mock_image_open.return_value = mock_image

            dataset = LEGOStepDataset(
                mock_filesystem["image_dir"],
                mock_filesystem["annotation_file"]
            )

            image, target = dataset[0]

            assert isinstance(target, dict)
            assert 'boxes' in target
            assert 'labels' in target
            assert 'image_id' in target
            assert isinstance(target['boxes'], torch.Tensor)
            assert isinstance(target['labels'], torch.Tensor)
            assert isinstance(target['image_id'], torch.Tensor)

    def test_load_annotations_file_not_found(self, mock_filesystem):
        """Test handling of missing annotation file"""
        mock_filesystem["annotation_file"].open.side_effect = FileNotFoundError
        dataset = LEGOStepDataset(
            mock_filesystem["image_dir"],
            mock_filesystem["annotation_file"]
        )

        assert len(dataset.annotations) == 0

    def test_load_annotations_invalid_json(self, mock_filesystem):
        """Test handling of invalid JSON"""
        with patch('json.load', side_effect=json.JSONDecodeError("Invalid JSON", "", 0)):
            dataset = LEGOStepDataset(
                mock_filesystem["image_dir"],
                mock_filesystem["annotation_file"]
            )

            assert len(dataset.annotations) == 0

    def test_empty_boxes_and_labels(self, mock_filesystem):
        """Test handling of empty boxes and labels"""
        empty_annotations = {
            "data": [{
                "image_path": "12345_1_page1.jpg",
                "boxes": [],
                "labels": []
            }]
        }

        with patch('json.load', return_value=empty_annotations):
            dataset = LEGOStepDataset(
                mock_filesystem["image_dir"],
                mock_filesystem["annotation_file"]
            )

            assert len(dataset.annotations) == 1
            assert torch.equal(dataset.annotations[0]['boxes'], torch.zeros((0, 4), dtype=torch.float32))
            assert torch.equal(dataset.annotations[0]['labels'], torch.zeros((0,), dtype=torch.int64))