# test_json_merger.py
import pytest
from unittest.mock import patch, mock_open, MagicMock
import json
from pathlib import Path
from src.utils.json_merger import merge_json_files


@pytest.fixture
def globals():
    # Setup mock files in input directory
    mock_file1 = MagicMock(spec=Path)
    mock_file1.name = "file1.json"
    mock_file1.suffix = ".json"
    mock_file2 = MagicMock(spec=Path)
    mock_file2.name = "file2.json"
    mock_file2.suffix = ".json"

    return [mock_file1, mock_file2]


@pytest.fixture
def mock_filesystem():
    """Fixture to mock all filesystem-related operations"""

    with patch("pathlib.Path") as mock_path_class:
        # Setup mock for input folder
        mock_input_folder = MagicMock(spec=Path)
        mock_input_folder.iterdir.return_value = []

        # Setup mock for output path
        mock_output_path = MagicMock(spec=Path)
        mock_output_path.parent = MagicMock(spec=Path)
        mock_output_path.exists.return_value = False

        # Create a mock for file paths that will be created
        mock_file_path = MagicMock(spec=Path)

        # Setup mock for path joining
        mock_input_folder.__truediv__.return_value = mock_file_path

        with patch("builtins.open", new_callable=mock_open) as mock_file:
            yield {
                "input_folder": mock_input_folder,
                "output_path": mock_output_path,
                "file_path": mock_file_path,
                "path_class": mock_path_class,
                "file": mock_file,
            }


def test_new_merge_with_no_existing_output(mock_filesystem, globals):
    mock_filesystem["input_folder"].iterdir.return_value = [
        globals[0],
        globals[1],
    ]
    mock_filesystem["output_path"].exists.return_value = False

    def mock_json_load():
        yield [{"key1": "value1"}]
        yield [{"key2": "value2"}]

    mock_json_load_generator = mock_json_load()

    # Configure mock file reads
    mock_filesystem[
        "file"
    ].return_value.__enter__.return_value.read.side_effect = lambda: json.dumps(
        next(mock_json_load_generator)
    )

    # Run the function with mock Path objects
    result = merge_json_files(
        mock_filesystem["input_folder"], mock_filesystem["output_path"]
    )

    # Verify results
    assert result == [globals[0].name, globals[1].name]

    # Verify directory creation
    mock_filesystem["output_path"].parent.mkdir.assert_called_once_with(
        parents=True, exist_ok=True
    )

    # Collect all written data
    write_calls = mock_filesystem[
        "file"
    ].return_value.__enter__.return_value.write.call_args_list
    written_json = "".join(call[0][0] for call in write_calls)
    written_data = json.loads(written_json)

    assert written_data["data"] == [{"key1": "value1"}, {"key2": "value2"}]
    assert set(written_data["processed_files"]) == {
        globals[0].name,
        globals[1].name,
    }


def test_merge_with_existing_output(mock_filesystem, globals):
    mock_filesystem["input_folder"].iterdir.return_value = [globals[1]]
    mock_filesystem["output_path"].exists.return_value = True

    # Setup mock file contents
    existing_output = {
        "data": [{"key1": "value1"}],
        "processed_files": [globals[0].name],
    }

    def mock_json_load():
        yield existing_output
        yield [{"key2": "value2"}]

    mock_json_load_generator = mock_json_load()

    # Configure mock file reads
    mock_filesystem[
        "file"
    ].return_value.__enter__.return_value.read.side_effect = lambda: json.dumps(
        next(mock_json_load_generator)
    )

    # Run the function with mock Path objects
    result = merge_json_files(
        mock_filesystem["input_folder"], mock_filesystem["output_path"]
    )

    # Verify results
    assert result == [globals[1].name]

    # Collect all written data
    write_calls = mock_filesystem[
        "file"
    ].return_value.__enter__.return_value.write.call_args_list
    written_json = "".join(call[0][0] for call in write_calls)
    written_data = json.loads(written_json)

    assert written_data["data"] == [{"key1": "value1"}, {"key2": "value2"}]
    assert set(written_data["processed_files"]) == {globals[0].name, globals[1].name}


def test_merge_with_invalid_json(mock_filesystem, globals):

    mock_filesystem["input_folder"].iterdir.return_value = [
        globals[0],
        globals[1],
    ]
    mock_filesystem["output_path"].exists.return_value = False

    read_count = 0

    def mock_json_load(f):
        nonlocal read_count
        if read_count == 0:
            read_count = 1
            return [{"key1": "value1"}]
        raise json.JSONDecodeError("Invalid JSON", "doc", 0)

    # Configure mock file reads
    mock_filesystem[
        "file"
    ].return_value.__enter__.return_value.read.side_effect = lambda: json.dumps(
        mock_json_load(mock_filesystem["file"])
    )

    # Run the function with mock Path objects
    result = merge_json_files(
        mock_filesystem["input_folder"], mock_filesystem["output_path"]
    )

    # Verify results
    assert result == [globals[0].name]

    # Collect all written data
    write_calls = mock_filesystem[
        "file"
    ].return_value.__enter__.return_value.write.call_args_list
    written_json = "".join(call[0][0] for call in write_calls)
    written_data = json.loads(written_json)

    assert written_data["data"] == [{"key1": "value1"}]
    assert set(written_data["processed_files"]) == {globals[0].name}


def test_no_new_files_to_process(mock_filesystem, globals):
    mock_filesystem["input_folder"].iterdir.return_value = [globals[0]]
    mock_filesystem["output_path"].exists.return_value = True

    # Setup mock file contents
    existing_output = {
        "data": [{"key1": "value1"}],
        "processed_files": [globals[0].name],
    }

    # Configure mock file reads
    mock_filesystem[
        "file"
    ].return_value.__enter__.return_value.read.side_effect = lambda: json.dumps(
        existing_output
    )

    # Run the function with mock Path objects
    result = merge_json_files(
        mock_filesystem["input_folder"], mock_filesystem["output_path"]
    )

    # Verify results
    assert result == []

    # Verify no new writes occurred
    mock_filesystem[
        "file"
    ].return_value.__enter__.return_value.write.assert_not_called()
