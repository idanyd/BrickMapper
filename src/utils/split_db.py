import os
import random
from pathlib import Path


def create_dataset_splits(
    source_dir,
    output_dir,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    random_seed=42,
):
    """
    Creates text files listing the files for training, validation, and test sets.

    Args:
        source_dir (Path): Path to source directory containing the files
        output_dir (Path): Directory where the split text files will be created
        train_ratio (float): Proportion of files for training (default: 0.6)
        val_ratio (float): Proportion of files for validation (default: 0.2)
        test_ratio (float): Proportion of files for testing (default: 0.2)
        random_seed (int): Random seed for reproducibility (default: 42)
    """
    # Verify ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Ratios must sum to 1")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of all files
    all_files = [f for f in source_dir.glob("*") if f.is_file()]

    # Set random seed for reproducibility
    random.seed(random_seed)
    random.shuffle(all_files)

    # Calculate split indices
    total_files = len(all_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    # Split files
    splits = {
        "train": all_files[:train_end],
        "val": all_files[train_end:val_end],
        "test": all_files[val_end:],
    }

    # Write file lists to text files
    for split_name, files in splits.items():
        output_file = output_dir / f"{split_name}.txt"
        with output_file.open("w", encoding="utf-8") as f:
            for file_path in files:
                # Write relative path to make the output more portable
                relative_path = os.path.relpath(
                    file_path, source_dir.parents[3]
                )
                f.write(f"{relative_path}\n")

    # Print summary
    print(f"Split files created in {output_dir}:")
    for split_name, files in splits.items():
        print(
            f"{split_name}.txt: {len(files)} files ({len(files)/total_files:.1%})"
        )


if __name__ == "__main__":
    # Example usage
    source_directory = Path("data/training/annotations/images")
    output_directory = Path("data/training/annotations")

    create_dataset_splits(source_directory, output_directory)
