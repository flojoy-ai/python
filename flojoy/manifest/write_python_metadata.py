import fnmatch
import json
import os

import pathlib

NODES_PATH = pathlib.Path(__file__).parent.parent.joinpath("nodes")

# The pattern to match for Python files
pattern = "*.py"

badbadnotgood = ["VCTR.py", "__init__.py", ".DS_Store"]
ignore_folders = [
    "venv",
]


def get_node_files():
    # List to store the file paths
    file_paths: list[str] = []
    for root, dirs, files in os.walk(NODES_PATH):
        for file in files:
            # Check if the file matches the pattern
            if any(
                folder_name in os.path.join(root, file)
                for folder_name in ignore_folders
            ):
                continue
            if fnmatch.fnmatch(file, pattern):
                # If it matches, add the full path to the list
                if file not in badbadnotgood:
                    file_paths.append(os.path.join(root, file))
    return file_paths


def write_metadata(out_path: str):
    node_files = get_node_files()
    function_dict: dict[str, dict[str, str]] = dict()
    for single_file in node_files:
        with open(single_file) as f:
            function_dict[os.path.basename(single_file)] = {
                "metadata": f.read(),
                "path": single_file,
            }

    s = json.dumps(obj=function_dict, indent=2)
    with open(out_path, "w") as out_file:
        out_file.write(s)
        out_file.write("\n")
