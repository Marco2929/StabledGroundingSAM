import os
import argparse
from typing import List


def positive_float(value: float) -> float:
    """
    Converts the input value to a positive float.

    :param value: Input value to be converted
    :type value: float

    :return: Converted positive float value
    :rtype: float

    :raises argparse.ArgumentTypeError: If the input value is not a valid positive float
    """

    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive float value" % value)
    return fvalue


def build_folder_structure() -> None:
    """
    Builds the folder structure for the output of the program.
    """

    main_folder = "output"
    sub_folders = ["segmented_pictures", "finished_dataset"]

    # Create the main folder
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    # Create sub-folders inside the main folder
    for folder_name in sub_folders:
        folder_path = os.path.join(main_folder, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Create "images" and "annotations" folders inside the "finished_dataset" folder
    finished_dataset_path = os.path.join(main_folder, "finished_dataset")
    images_folder_path = os.path.join(finished_dataset_path, "images")
    annotations_folder_path = os.path.join(finished_dataset_path, "annotations")

    if not os.path.exists(images_folder_path):
        os.makedirs(images_folder_path)

    if not os.path.exists(annotations_folder_path):
        os.makedirs(annotations_folder_path)


def get_class_names(path: str) -> List[str]:
    """
    Reads a file at the specified path and retrieves a list of class names.

    :param path: Path to the file containing class names
    :type path: str

    :return: List of class names
    :rtype: List[str]
    """
    # Open the file
    with open(path, 'r') as file:
        # Read the contents
        contents = file.read()

    # Split the names using a comma as delimiter
    names = contents.split(',')

    # Remove any leading or trailing whitespace from each name
    names = [name.strip() for name in names]

    return names
