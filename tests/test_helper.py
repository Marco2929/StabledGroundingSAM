import unittest
import os
import argparse

from ..helper import positive_float, build_folder_structure, get_class_names


class TestProgram(unittest.TestCase):

    def test_positive_float(self):
        self.assertEqual(positive_float(3.14), 3.14)
        self.assertEqual(positive_float("5.0"), 5.0)

        with self.assertRaises(argparse.ArgumentTypeError):
            positive_float(-2.5)
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_float("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_float("not_a_float")

    def test_build_folder_structure(self):
        # Test if the folder structure is built correctly
        build_folder_structure()

        self.assertTrue(os.path.exists("output"))
        self.assertTrue(os.path.exists(os.path.join("output", "segmented_pictures")))
        self.assertTrue(os.path.exists(os.path.join("output", "finished_dataset")))
        self.assertTrue(os.path.exists(os.path.join("output", "finished_dataset", "images")))
        self.assertTrue(os.path.exists(os.path.join("output", "finished_dataset", "annotations")))

    def test_get_class_names(self):
        # Create a temporary file with class names
        with open("class_names.txt", "w") as file:
            file.write("class1, class2, class3")

        # Test if the class names are read correctly
        class_names = get_class_names("class_names.txt")
        self.assertEqual(class_names, ["class1", "class2", "class3"])

        # Clean up the temporary file
        os.remove("class_names.txt")

        # Test error response for non-existent file
        with self.assertRaises(FileNotFoundError):
            get_class_names("non_existent_file.txt")

        # Test error response for empty file
        with open("empty_file.txt", "w"):
            pass
        with self.assertRaises(IndexError):
            get_class_names("empty_file.txt")
        os.remove("empty_file.txt")


if __name__ == "__main__":
    unittest.main()
