import os
from segment_anything import sam_model_registry, SamPredictor
import torch
from GroundingDINO.groundingdino.util.inference import Model
from diffusers import StableDiffusionImg2ImgPipeline


def init_models(HOME, device):
    print("Initializing models..")
    MODEL_TYPE = "vit_h"

    # Initialize Stable Diffusion
    model_id_or_path = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)

    pipe = pipe.to(device)

    # Initialize Grounding DINO
    GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))

    GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
    print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))

    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # Initialize SAM
    CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
    print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=device)
    sam_predictor = SamPredictor(sam)


    return pipe, grounding_dino_model, sam_predictor