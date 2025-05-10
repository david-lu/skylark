import os
import sys
from typing import Tuple, List, Optional, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.evf_sam2_video import EvfSam2Model


# Configuration constants
PRETRAINED_MODEL_PATH = "evf-sam2"  # Path to the pretrained model
OUTPUT_PATH = "./infer"  # Path to save output visualizations
IMAGE_SIZE = 224  # Size to resize images to
FRAMES_PATH = "assets/zebra.jpg"  # Path to input frames
PROMPT = "zebra top left"  # Text prompt for segmentation


def beit3_preprocess(
    x: np.ndarray,
    img_size: int = 224,
) -> torch.Tensor:
    '''
    preprocess for BEIT-3 model.
    input: ndarray
    output: torch.Tensor
    '''
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(x)


def init_models(
    pretrained_model_name_or_path: str,
    precision: str = "fp16",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> Tuple[PreTrainedTokenizer, EvfSam2Model]:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        padding_side="right",
        use_fast=False,
    )

    torch_dtype = torch.float32
    if precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp16":
        torch_dtype = torch.half

    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
            }
        )
    elif load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    # Always use sam2 model
    model = EvfSam2Model.from_pretrained(
        pretrained_model_name_or_path, low_cpu_mem_usage=True, **kwargs
    )

    if (not load_in_4bit) and (not load_in_8bit):
        model = model.cuda()
    model.eval()

    return tokenizer, model


def process_frames(
    tokenizer: PreTrainedTokenizer,
    model: EvfSam2Model,
    input_frames_path: str,
    output_frames_path: str,
    prompt: str,
    image_size: int = 224,
) -> None:
    # clarify IO
    if not os.path.exists(input_frames_path):
        print("File not found in {}".format(input_frames_path))
        exit()

    os.makedirs(output_frames_path, exist_ok=True)
    
    frame_files = os.listdir(input_frames_path)
    frame_files.sort()

    # preprocess    
    image_np = cv2.imread(os.path.join(input_frames_path, frame_files[0]))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # original_size_list = [image_np.shape[:2]]

    image_beit = beit3_preprocess(image_np, image_size).to(dtype=model.dtype, device=model.device)

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=model.device)

    # infer
    output = model.inference(
        input_frames_path,
        image_beit.unsqueeze(0),
        input_ids,
        # original_size_list=original_size_list,
    )
    # save visualization
    for i, file in enumerate(frame_files):
        img = cv2.imread(os.path.join(input_frames_path, file))
        out = img + np.array([0,0,128]) * output[i][1].transpose(1,2,0)
        cv2.imwrite(os.path.join(output_frames_path, file), out)


def init_config() -> None:
    """Initialize CUDA configuration for optimal performance."""
    # use float16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


if __name__ == "__main__":
    # Initialize CUDA configuration
    init_config()
        
    # initialize model and tokenizer
    tokenizer, model = init_models(
        pretrained_model_name_or_path=PRETRAINED_MODEL_PATH,
    )
    
    # Process the images
    process_frames(
        tokenizer=tokenizer,
        model=model,
        input_frames_path=FRAMES_PATH,
        output_frames_path=OUTPUT_PATH,
        prompt=PROMPT,
        image_size=IMAGE_SIZE
    )