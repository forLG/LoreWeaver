import os
import json
import glob

def get_all_types(data, types_set):
    """Recursively traverse JSON data to find 'type' fields."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "type" and isinstance(value, str):
                types_set.add(value)
            get_all_types(value, types_set)
    elif isinstance(data, list):
        for item in data:
            get_all_types(item, types_set)

def main():
    # Define the directory path
    data_dir = "data_src/adventure"
    
    # Find all adventure-*.json files
    pattern = os.path.join(data_dir, "adventure-*.json")
    files = glob.glob(pattern)
    
    all_types = set()
    
    print(f"Scanning {len(files)} files...")
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                get_all_types(data, all_types)
                print(f"Processed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print("\nAll found 'type' values:")
    print("-" * 30)
    for t in sorted(list(all_types)):
        print(t)

if __name__ == "__main__":
    main()