from trainer import LEGOModelTrainer
from pathlib import Path
import logging
import sys

logger = logging.getLogger(__name__)

def main():

    try:
        trainer = LEGOModelTrainer(
            images_dir=Path("data/images"),
            annotations_dir=Path("data/annotations"),
            model_save_path=Path("models/lego_detector_cuda.pth")
        )
        trainer.train()
        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)    

if __name__ == "__main__":
    from logger import setup_logging
    setup_logging(logging.DEBUG)

    main()
