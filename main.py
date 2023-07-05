from init_models import init_models
from typing import List
import numpy as np
from segment_anything import SamPredictor
import cv2
from annotate import BoxAnnotator, MaskAnnotator
from PIL import Image
import argparse
import os
import supervision as sv
from helper import build_folder_structure, positive_float, get_class_names


def parse_args():
    parser = argparse.ArgumentParser(description='Process an image with a quote overlay')

    parser.add_argument('image_path', type=str, help='Path to the input image file')
    parser.add_argument('class_path', type=str, help='txt file of which objects appear in image')
    parser.add_argument('number_of_pictures', type=int, help='how many pictures should be generated')

    parser.add_argument('-dp', '--diffusion_prompt', type=str, help='might be useful if generated image differs to classes')
    parser.add_argument('-bt', '--box_threshold', type=float,
                        help='Threshold for the bounding box detection (default: 0.5)')
    parser.add_argument('-tt', '--text_threshold', type=float,
                        help='Threshold for the text detection (default: 0.7)')
    parser.add_argument('-s', '--strength', type=positive_float,
                        help='Strength of stable pipe')
    parser.add_argument('-gs', '--guidance_scale', type=positive_float,
                        help='Guidance scale of stable pipe')
    parser.add_argument('-min', '--min_image_area_percentage', type=positive_float,
                        help='Minimum area a mask should have in the dataset')
    parser.add_argument('-max', '--max_image_area_percentage', type=positive_float,
                        help='Maximum area a mask should have in the dataset')
    parser.add_argument('-app', '--approximation_percentage', type=positive_float,
                        help='How sharp the edges of the masks are the higher the sharper')

    args = parser.parse_args()
    return args


def enhance_class_name(class_names: List[str]) -> List[str]:
    """
    Enhances the class names by adding a prefix and pluralizing them.

    :param class_names: List of class names to be enhanced
    :type class_names: List[str]
    :return: List of enhanced class names
    :rtype: List[str]
    """

    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """
    Segments the specified image using the provided SamPredictor.

    :param sam_predictor: Instance of SamPredictor used for segmentation
    :type sam_predictor: SamPredictor
    :param image: Input image as a NumPy array
    :type image: np.ndarray
    :param xyxy: Bounding box coordinates in the format [x1, y1, x2, y2]
    :type xyxy: np.ndarray
    :return: Segmented masks corresponding to the bounding boxes
    :rtype: np.ndarray
    """

    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


class GenerateDataset:
    """
    Initializes an instance of the GenerateDataset class with the provided arguments.

    :param image_path: Path to the input image file
    :type image_path: str
    :param class_path: Path to the text file containing the objects that appear in the image
    :type class_path: str
    :param number_of_pictures: Number of pictures to be generated
    :type number_of_pictures: int
    :param diffusion_prompt: Optional prompt to use if the generated image differs from the expected classes
    :type diffusion_prompt: str
    :param box_threshold: Threshold for the bounding box detection (default: 0.35)
    :type box_threshold: float
    :param text_threshold: Threshold for the text detection (default: 0.25)
    :type text_threshold: float
    :param strength: Strength of stable pipe (default: 0.75)
    :type strength: float
    :param guidance_scale: Guidance scale of stable pipe (default: 7.5)
    :type guidance_scale: float
    :param min_image_area_percentage: Minimum area a mask should have in the dataset (default: 0.002)
    :type min_image_area_percentage: float
    :param max_image_area_percentage: Maximum area a mask should have in the dataset (default: 0.80)
    :type max_image_area_percentage: float
    :param approximation_percentage: How sharp the edges of the masks are; the higher the sharper (default: 0.75)
    :type approximation_percentage: float
    """
    def __init__(self,
                 image_path: str,
                 class_path: str,
                 number_of_pictures: int,
                 diffusion_prompt: str,
                 box_threshold: float = 0.35,
                 text_threshold: float = 0.25,
                 strength: float = 0.75,
                 guidance_scale: float = 7.5,
                 min_image_area_percentage: float = 0.002,
                 max_image_area_percentage: float = 0.80,
                 approximation_percentage: float = 0.75):
        self.image_path = image_path
        self.class_path = class_path
        self.number_of_pictures = number_of_pictures
        self.diffusion_prompt = diffusion_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.min_image_area_percentage = min_image_area_percentage
        self.max_image_area_percentage = max_image_area_percentage
        self.approximation_percentage = approximation_percentage

    def main(self):
        # get current path of program
        home = os.path.dirname(os.path.abspath(__file__))

        # load models
        stable_pipe, grounding_dino_model, sam_predictor = init_models(home)

        # build folder structure
        build_folder_structure()

        # load image
        image = Image.open(self.image_path).convert("RGB")

        CLASSES = get_class_names(self.class_path)

        if self.diffusion_prompt is None:
            self.diffusion_prompt = " ".join(CLASSES)  # Convert list of classes to a space-separated string
            self.diffusion_prompt = self.diffusion_prompt.replace(" ", ", ")  # Add commas between the class names

        print("Starting image generation and segmentation..")

        images = {}
        annotations = {}

        images_extensions = ['jpg', 'jpeg', 'png']

        for i in range(self.number_of_pictures):

            stabled_image = stable_pipe(prompt=self.diffusion_prompt, image=image, strength=self.strength,
                                 guidance_scale=self.guidance_scale).images

            stabled_image_path = f"{home}/output/finished_dataset/images/generated_image_{i + 1}.png"

            stabled_image[0].save(stabled_image_path)

            # Load the generated image
            generated_image = cv2.imread(stabled_image_path)

            detections = grounding_dino_model.predict_with_classes(
                image=image,
                classes=enhance_class_name(class_names=CLASSES),
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold
            )
            detections = detections[detections.class_id != None]
            detections.mask = segment(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )
            images[i + 1] = image
            annotations[i + 1] = detections

            # Annotate image with detections
            box_annotator = BoxAnnotator()
            mask_annotator = MaskAnnotator()
            labels = [
                f"{CLASSES[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _
                in detections]
            annotated_image = mask_annotator.annotate(scene=generated_image.copy(), detections=detections)
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

            # Save the annotated image
            cv2.imwrite(f"{home}/output/segmented_pictures/generated_image_{i + 1}.png", annotated_image)

        sv.DetectionDataset(
            classes=CLASSES,
            images=images,
            annotations=annotations
        ).as_yolo(
            annotations_directory_path=f"{home}/output/finished_dataset/annotations",
            min_image_area_percentage=self.min_image_area_percentage,
            max_image_area_percentage=self.max_image_area_percentage,
            approximation_percentage=self.approximation_percentage
        )


if __name__ == '__main__':
    args = parse_args()

    # Create an instance of the GenerateDataset class with the specified arguments
    generate = GenerateDataset(
                               args.image_path,
                               args.class_path,
                               args.number_of_pictures,
                               args.diffusion_prompt,
                               args.box_threshold,
                               args.text_threshold,
                               args.strength,
                               args.guidance_scale,
                               args.min_image_area_percentage,
                               args.max_image_area_percentage,
                               args.approximation_percentage)

    generate.main()




