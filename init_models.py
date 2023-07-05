import os
import torch
from segment_anything import sam_model_registry, SamPredictor
from GroundingDINO.groundingdino.util.inference import Model
from diffusers import StableDiffusionImg2ImgPipeline

# Check CUDA availability
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"You're currently using {device}")


def init_models(home):
    """
    Initializes the models for the application.

    :param home: Path to the home directory
    :type home: str

    :return: Tuple containing the initialized models (pipe, grounding_dino_model, sam_predictor)
    :rtype: Tuple[StableDiffusionImg2ImgPipeline, Model, SamPredictor]
    """

    print("Initializing models..")
    model_type = "vit_h"

    # Initialize Stable Diffusion
    stable_diffusion_model_id = "runwayml/stable-diffusion-v1-5"
    stable_diffusion_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(stable_diffusion_model_id, torch_dtype=torch.float16)
    stable_diffusion_pipe = stable_diffusion_pipe.to(device)

    # Initialize Grounding DINO
    grounding_dino_config_path = os.path.join(home, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    grounding_dino_checkpoint_path = os.path.join(home, "weights", "groundingdino_swint_ogc.pth")

    print(grounding_dino_config_path, "; exist:", os.path.isfile(grounding_dino_config_path))
    print(grounding_dino_checkpoint_path, "; exist:", os.path.isfile(grounding_dino_checkpoint_path))

    grounding_dino_model = Model(model_config_path=grounding_dino_config_path, model_checkpoint_path=grounding_dino_checkpoint_path)

    # Initialize SAM
    sam_checkpoint_path = os.path.join(home, "weights", "sam_vit_h_4b8939.pth")
    print(sam_checkpoint_path, "; exist:", os.path.isfile(sam_checkpoint_path))

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path).to(device='cpu')
    sam_predictor = SamPredictor(sam)

    return stable_diffusion_pipe, grounding_dino_model, sam_predictor
