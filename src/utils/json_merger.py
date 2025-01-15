# json_merger.py
import json

# from pathlib import Path


def merge_json_files(input_folder, output_path):
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing merged data and processed files if they exist
    if output_path.exists():
        with open(output_path, "r") as f:
            output_data = json.load(f)
            merged_data = output_data.get("data", [])
            processed_files = set(output_data.get("processed_files", []))
    else:
        merged_data = []
        processed_files = set()

    # Get all JSON files in the input folder
    json_files = [
        f.name for f in input_folder.iterdir() if f.suffix == ".json"
    ]
    new_files_processed = []

    # Process each unprocessed JSON file
    for file_name in json_files:
        if file_name not in processed_files:
            file_path = input_folder / file_name
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Add the data to merged_data
                merged_data.extend(data)

                processed_files.add(file_name)
                new_files_processed.append(file_name)

            except json.JSONDecodeError as e:
                print(f"Error processing {file_name}: {str(e)}")
            except Exception as e:
                print(f"Unexpected error processing {file_name}: {str(e)}")

    # Save the merged data if new files were processed
    if new_files_processed:
        output_data = {
            "processed_files": list(processed_files),
            "data": merged_data,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)

    return new_files_processed
