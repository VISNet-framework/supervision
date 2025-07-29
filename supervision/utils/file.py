from __future__ import annotations

import datetime
import json
from pathlib import Path

import natsort
import numpy as np
import yaml


class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ExtendedJSONEncoder(json.JSONEncoder):
    """Special json encoder for numpy types, paths and datetimes"""

    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def list_files_with_extensions(
    directory: str | Path, extensions: list[str] | None = None
) -> list[Path]:
    """
    list files in a directory with specified extensions or
        all files if no extensions are provided.

    Args:
        directory (Union[str, Path]): The directory path as a string or Path object.
        extensions (Optional[list[str]]): A list of file extensions to filter.
            Default is None, which lists all files.

    Returns:
        (list[Path]): A list of Path objects for the matching files.

    Examples:
        ```python
        import supervision as sv

        # list all files in the directory
        files = sv.list_files_with_extensions(directory='my_directory')

        # list only files with '.txt' and '.md' extensions
        files = sv.list_files_with_extensions(
            directory='my_directory', extensions=['txt', 'md'])
        ```
    """

    directory = Path(directory)
    files_with_extensions = []

    if extensions is not None:
        for ext in extensions:
            files_with_extensions.extend(directory.glob(f"*.{ext}"))
    else:
        files_with_extensions.extend(directory.glob("*"))

    return files_with_extensions


def list_files_with_extensions_recursively(
    directory: str | Path, extensions: list[str] | None = None
) -> list[Path]:
    """
    list files in a directory and its subdirectories with specified extensions
        or all files if no extensions are provided.

    Args:
        directory (Union[str, Path]): The directory path as a string or Path object.
        extensions (Optional[list[str]]): A list of file extensions to filter.
            Default is None, which lists all files.

    Returns:
        (list[Path]): A list of Path objects for the matching files.
    """
    directory = Path(directory)
    files_with_extensions = []

    if extensions is not None:
        for ext in extensions:
            files_with_extensions.extend(directory.rglob(f"*.{ext}"))
    else:
        files_with_extensions.extend(directory.rglob("*"))

    return files_with_extensions


def find_valid_images_and_annotations(
    images_directory_path: Path,
    annotation_path: Path,
    images_extentions=["jpg", "jpeg", "png", "tiff", "tif"],
    annotation_extentions=["json"],
) -> tuple[list[Path], list[Path]]:
    """
    Finds and matches valid image files and their corresponding annotation files

    Args:
        images_directory_path (Path): Path to the directory containing image files.
        annotation_path (Path): Path to the directory containing annotation files.
        images_extentions (list[str], optional): list of valid image file extensions.
        annotation_extentions (list[str], optional): list of valid annotation file ext.

    Returns:
        Tuple[list[Path], list[Path]]:
            - list of image file paths that have corresponding annotation files.
            - list of annotation file paths, sorted in natural order.
    """

    image_candidate_paths = list_files_with_extensions_recursively(
        directory=images_directory_path,
        extensions=images_extentions,
    )
    image_candidate_stems = [path.stem for path in image_candidate_paths]
    assert len(image_candidate_stems) == len(set(image_candidate_stems)), (
        "Image filenames must be unique"
    )

    annotation_paths = list_files_with_extensions_recursively(
        directory=annotation_path,
        extensions=annotation_extentions,
    )
    annotation_paths = natsort.natsorted(annotation_paths)

    image_paths = []
    for annotation_path in annotation_paths:
        # find the corresponding image path
        image_stem = annotation_path.stem
        image_path = image_candidate_paths[image_candidate_stems.index(image_stem)]
        image_paths.append(image_path)
    return image_paths, annotation_paths


def read_txt_file(file_path: str | Path, skip_empty: bool = False) -> list[str]:
    """
    Read a text file and return a list of strings without newline characters.
    Optionally skip empty lines.

    Args:
        file_path (Union[str, Path]): The file path as a string or Path object.
        skip_empty (bool): If True, skip lines that are empty or contain only
            whitespace. Default is False.

    Returns:
        list[str]: A list of strings representing the lines in the text file.
    """
    with open(str(file_path)) as file:
        if skip_empty:
            lines = [line.rstrip("\n") for line in file if line.strip()]
        else:
            lines = [line.rstrip("\n") for line in file]

    return lines


def save_text_file(lines: list[str], file_path: str | Path) -> None:
    """
    Write a list of strings to a text file, each string on a new line.

    Args:
        lines (list[str]): The list of strings to be written to the file.
        file_path (Union[str, Path]): The file path as a string or Path object.
    """
    with open(str(file_path), "w") as file:
        for line in lines:
            file.write(line + "\n")


def read_json_file(file_path: str | Path) -> dict:
    """
    Read a json file and return a dict.

    Args:
        file_path (Union[str, Path]): The file path as a string or Path object.

    Returns:
        dict: A dict of annotations information
    """
    with open(str(file_path)) as file:
        data = json.load(file)
    return data


def save_json_file(data: dict, file_path: str | Path, indent: int = 3) -> None:
    """
    Write a dict to a json file.

    Args:
        indent:
        data (dict): dict with unique keys and value as pair.
        file_path (Union[str, Path]): The file path as a string or Path object.
    """
    with open(str(file_path), "w") as fp:
        json.dump(data, fp, cls=NumpyJsonEncoder, indent=indent)


def read_yaml_file(file_path: str | Path) -> dict:
    """
    Read a yaml file and return a dict.

    Args:
        file_path (Union[str, Path]): The file path as a string or Path object.

    Returns:
        dict: A dict of content information
    """
    with open(str(file_path)) as file:
        data = yaml.safe_load(file)
    return data


def save_yaml_file(data: dict, file_path: str | Path) -> None:
    """
    Save a dict to a json file.

    Args:
        indent:
        data (dict): dict with unique keys and value as pair.
        file_path (Union[str, Path]): The file path as a string or Path object.
    """

    with open(str(file_path), "w") as outfile:
        yaml.dump(data, outfile, sort_keys=False, default_flow_style=None)
