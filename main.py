from init_models import init_models
from typing import List
import numpy as np
from segment_anything import SamPredictor
import cv2
from annotate import BoxAnnotator, MaskAnnotator
from PIL import Image
import argparse
import os
import torch
import supervision as sv
from helper import build_folder_structure, positive_float, get_class_names


def parse_args():
    parser = argparse.ArgumentParser(description='Process an image with a quote overlay')

    parser.add_argument('image_path', type=str, help='Path to the input image file')
    parser.add_argument('class_path', type=str, help='txt file of which objects appear in image')
    parser.add_argument('number_of_pictures', type=int, help='how many pictures should be generated')

    parser.add_argument('-dp', '--diffusion_prompt', type=str, help='might be useful if generated image differs to classes')
    parser.add_argument('-bt', '--box_threshold', type=float, default=0.35,
                        help='Threshold for the bounding box detection (default: 0.5)')
    parser.add_argument('-tt', '--text_threshold', type=float, default=0.25,
                        help='Threshold for the text detection (default: 0.7)')
    parser.add_argument('-s', '--strength', type=positive_float, default=0.75,
                        help='Strength of stable pipe')
    parser.add_argument('-gs', '--guidance_scale', type=positive_float, default=7.5,
                        help='Guidance scale of stable pipe')
    parser.add_argument('-min', '--min_image_area_percentage', type=positive_float, default=0.002,
                        help='Minimum area a mask should have in the dataset')
    parser.add_argument('-max', '--max_image_area_percentage', type=positive_float, default=0.80,
                        help='Maximum area a mask should have in the dataset')
    parser.add_argument('-app', '--approximation_percentage', type=positive_float, default=0.75,
                        help='How sharp the edges of the masks are the higher the sharper')

    args = parser.parse_args()
    return args


class GenerateDataset:
    def __init__(self, device, image_path, class_path, number_of_pictures, diffusion_prompt, box_threshold, text_threshold, strength,
                 guidance_scale, min_image_area_percentage, max_image_area_percentage, approximation_percentage):
        self.device = device
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

    def enhance_class_name(self, class_names: List[str]) -> List[str]:
        return [
            f"all {class_name}s"
            for class_name
            in class_names
        ]

    def segment(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
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

    def main(self):
        # get current path of program
        home = os.path.dirname(os.path.abspath(__file__))

        # load models
        stable_pipe, grounding_dino_model, sam_predictor = init_models(home, device)

        # build folder structure
        build_folder_structure()

        # load image
        image = Image.open(self.image_path).convert("RGB")

        CLASSES = get_class_names(self.class_path)

        if self.diffusion_prompt is None:
            self.diffusion_prompt = " ".join(CLASSES)  # Convert list of classes to a space-separated string
            self.diffusion_prompt = self.diffusion_prompt.replace(" ", ", ")  # Add commas between the class names
        print("Starting image generation and segmentation..")
        for i in range(self.number_of_pictures):
            images = stable_pipe(prompt=self.diffusion_prompt, image=image, strength=self.strength,
                                 guidance_scale=self.guidance_scale).images

            stabled_image_path = f"{home}/output/finished_dataset/images/generated_image_{i + 1}.png"

            images[0].save(stabled_image_path)

            # Load the generated image
            generated_image = cv2.imread(stabled_image_path)

            # Detect objects in the generated image
            detections = grounding_dino_model.predict_with_classes(
                image=generated_image,
                classes=self.enhance_class_name(class_names=CLASSES),
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold
            )

            # Convert detections to masks
            detections.mask = self.segment(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )

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

        images_extensions = ['jpg', 'jpeg', 'png']

        images = {}
        annotations = {}

        image_paths = sv.list_files_with_extensions(
            directory=f"{home}/output/segmented_pictures",
            extensions=images_extensions)
        print("Save dataset as yolo..")
        for image_path in image_paths:
            image_name = image_path.name
            image_path = str(image_path)
            image = cv2.imread(image_path)

            detections = grounding_dino_model.predict_with_classes(
                image=image,
                classes=self.enhance_class_name(class_names=CLASSES),
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold
            )
            detections = detections[detections.class_id != None]
            detections.mask = self.segment(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )
            images[image_name] = image
            annotations[image_name] = detections

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

    # Check Cuda Availability
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"You're currently using {device}")

    # Create an instance of the GenerateDataset class with the specified arguments
    generate = GenerateDataset(
                               device,
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




