import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch, MagicMock, mock_open
from src.lego_step_detector import LEGOStepDetector
from torch.utils.data import DataLoader

@pytest.fixture
def mock_filesystem():
    with patch("pathlib.Path") as mock_path_class:
        # Setup mock for model path
        mock_model_path = MagicMock(spec=Path)
        mock_model_path.parent = MagicMock(spec=Path)
        mock_model_path.exists.return_value = False

        # Setup mock for file operations
        mock_file_path = MagicMock(spec=Path)
        mock_file_path.exists.return_value = True

        # Setup path joining behavior
        mock_model_path.__truediv__.return_value = mock_file_path

        with patch("builtins.open", new_callable=mock_open) as mock_file:
            yield {
                "model_path": mock_model_path,
                "file_path": mock_file_path,
                "path_class": mock_path_class,
                "file": mock_file,
            }

@pytest.fixture
def mock_model():
    with patch('src.lego_step_detector.fasterrcnn_resnet50_fpn') as mock_fn:
        model = Mock()
        model.roi_heads.box_predictor.cls_score.in_features = 256
        mock_fn.return_value = model
        yield mock_fn

@pytest.fixture
@patch('torch.cuda.is_available', return_value=False)
@patch('torch.device', return_value=torch.device('cpu'))
@patch('src.lego_step_detector.fasterrcnn_resnet50_fpn')
def detector_cpu(mock_fn, mock_device, mock_cuda):
    mock_fn.to.return_value = None
    detector = LEGOStepDetector()
    return detector

@pytest.fixture
@patch('torch.cuda.is_available', return_value=True)
@patch('torch.device', return_value=torch.device('cuda'))
@patch('src.lego_step_detector.fasterrcnn_resnet50_fpn')
def detector_cuda(mock_fn, mock_device, mock_cuda):
    mock_fn.to.return_value = None
    detector = LEGOStepDetector()
    return detector

@pytest.fixture
def sample_image():
    return Image.new('RGB', (100, 100))

@pytest.fixture
def mock_dataset():
    dataset = MagicMock(spec=['__len__', '__getitem__'])
    dataset.__len__.return_value = 10
    
    # Mock the dataset's __getitem__ method to return proper training data
    def getitem(idx):
        return (
            torch.rand(3, 100, 100),  # Mock image tensor
            {   # Mock target dict
                'boxes': torch.tensor([[0, 0, 1, 1]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),
                'image_id': torch.tensor([idx])
            }
        )
    dataset.__getitem__.side_effect = getitem

    return dataset

@pytest.fixture
def mock_loss_dict():
    return {
        'loss_classifier': torch.tensor(1.0, requires_grad=True),
        'loss_box_reg': torch.tensor(0.5, requires_grad=True),
        'loss_objectness': torch.tensor(0.3, requires_grad=True),
        'loss_rpn_box_reg': torch.tensor(0.2, requires_grad=True)
    }


class TestLEGOStepDetector:
    def test_initialization(self, detector_cpu):
        
            """Test detector initialization"""
            assert hasattr(detector_cpu, 'model')
            assert hasattr(detector_cpu, 'device')

    @patch('torch.optim.SGD')
    def test_train(self, mock_optimizer, detector_cpu, mock_dataset):
        """Test training method"""
        # Create a mock optimizer instance
        mock_optimizer_instance = Mock()
        mock_optimizer.return_value = mock_optimizer_instance

        # Mock the model's forward pass to return a loss dict
        detector_cpu.model.return_value = {
            'loss_classifier': torch.tensor(1.0, requires_grad=True),
            'loss_box_reg': torch.tensor(0.5, requires_grad=True),
            'loss_objectness': torch.tensor(0.3, requires_grad=True),
            'loss_rpn_box_reg': torch.tensor(0.2, requires_grad=True)
        }

        # Create a real DataLoader with the mock dataset
        # This ensures proper iteration behavior
        DataLoader(
            mock_dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=detector_cpu._collate_fn
        )

        # Run training
        detector_cpu.train(mock_dataset, num_epochs=1)

        # Verify that optimizer was created with correct parameters
        mock_optimizer.assert_called_once()

        # Verify that optimizer steps were taken
        assert mock_optimizer_instance.zero_grad.call_count > 0
        assert mock_optimizer_instance.step.call_count > 0

        # Verify that the model was put into training mode
        detector_cpu.model.train.assert_called_once()

    def test_save_model(self, detector_cpu, mock_filesystem):
        """Test model saving"""
        with patch('torch.save') as mock_save:
            detector_cpu.save_model(mock_filesystem["model_path"])

            # Verify directory creation was attempted
            mock_filesystem["model_path"].parent.mkdir.assert_called_once_with(
                parents=True, exist_ok=True
            )

            # Verify model was saved
            mock_save.assert_called_once()

    def test_load_model(self, detector_cpu, mock_filesystem):
        """Test model loading"""
        with patch('torch.load') as mock_load:
            detector_cpu.load_model(mock_filesystem["model_path"])
            mock_load.assert_called_once_with(mock_filesystem["model_path"])

    def test_load_model_error(self, detector_cpu, mock_filesystem):
        """Test model loading error handling"""
        with patch('torch.load', side_effect=Exception("Load error")):
            with pytest.raises(Exception):
                detector_cpu.load_model(mock_filesystem["model_path"])

    def test_detect_steps(self, detector_cpu, sample_image):
        """Test step detection"""
        mock_predictions = [{
            'boxes': torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
            'labels': torch.tensor([1, 2]),
            'scores': torch.tensor([0.9, 0.8])
        }]
        detector_cpu.model.return_value = mock_predictions

        results = detector_cpu.detect_steps(sample_image)

        assert isinstance(results, list)
        assert len(results) > 0
        assert 'bbox' in results[0]
        assert 'step_number' in results[0]

    def test_collate_fn(self):
        """Test collate function"""
        batch = [
            (torch.rand(3, 32, 32), {'boxes': torch.tensor([[0, 0, 1, 1]])}),
            (torch.rand(3, 32, 32), {'boxes': torch.tensor([[1, 1, 2, 2]])}),
        ]
        result = LEGOStepDetector._collate_fn(batch)
        assert len(result) == 2
        assert len(result[0]) == 2

    def test_find_closest_step(self):
        """Test finding closest step"""
        number_box = np.array([10, 10, 20, 20])
        steps = {
            0: {'bbox': np.array([0, 0, 10, 10])},
            1: {'bbox': np.array([50, 50, 60, 60])}
        }
        result = LEGOStepDetector._find_closest_step(number_box, steps)
        assert result == 0

    def test_extract_number_region(self, detector_cpu, sample_image):
        """Test number region extraction"""
        bbox = np.array([10, 10, 20, 20])
        result = detector_cpu._extract_number_region(sample_image, bbox)
        assert isinstance(result, Image.Image)

    def test_extract_number_region_error(self, detector_cpu, sample_image):
        """Test number region extraction error handling"""
        bbox = np.array([-10, -10, -20, -20])  # Invalid bbox
        result = detector_cpu._extract_number_region(sample_image, bbox)
        assert result is None

    def test_process_number(self, detector_cpu, sample_image):
        """Test number processing"""
        result = detector_cpu._process_number(sample_image)
        assert result is None

    @pytest.mark.integration
    def test_full_detection_pipeline(self, detector_cpu, sample_image):
        """Integration test for full detection pipeline"""
        mock_predictions = [{
            'boxes': torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
            'labels': torch.tensor([1, 2]),
            'scores': torch.tensor([0.9, 0.8])
        }]
        detector_cpu.model.return_value = mock_predictions

        results = detector_cpu.detect_steps(sample_image)

        assert isinstance(results, list)
        assert len(results) > 0
        for result in results:
            assert 'bbox' in result
            assert 'step_number' in result
            assert isinstance(result['bbox'], tuple)
            assert len(result['bbox']) == 4

    def test_train_single_batch(self, detector_cpu, mock_dataset, mock_loss_dict):
        """Test training behavior for a single batch"""
        with patch('torch.optim.SGD') as mock_optimizer:
            mock_opt_instance = Mock()
            mock_optimizer.return_value = mock_opt_instance

            detector_cpu.model.return_value = mock_loss_dict

            detector_cpu.train(mock_dataset, num_epochs=1)

            # Verify training loop behavior
            assert mock_opt_instance.zero_grad.call_count > 0
            assert mock_opt_instance.step.call_count > 0
            detector_cpu.model.train.assert_called_once()

    def test_train_multiple_epochs(self, detector_cpu, mock_dataset, mock_loss_dict):
        """Test training behavior across multiple epochs"""
        num_epochs = 3
        with patch('torch.optim.SGD') as mock_optimizer:
            mock_opt_instance = Mock()
            mock_optimizer.return_value = mock_opt_instance

            detector_cpu.model.return_value = mock_loss_dict

            detector_cpu.train(mock_dataset, num_epochs=num_epochs)

            # Verify multiple epochs
            expected_steps = num_epochs * (len(mock_dataset) // 2)  # batch_size=2
            assert mock_opt_instance.step.call_count == expected_steps


class TestDeviceHandling:

    def test_gpu_initialization(self, detector_cuda):
        """Test initialization with GPU available"""
        assert str(detector_cuda.device) == 'cuda'

    def test_cpu_initialization(self, detector_cpu):
        """Test initialization with only CPU available"""
        detector = LEGOStepDetector()
        assert str(detector_cpu.device) == 'cpu'