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


def beit3_preprocess(x: np.ndarray, img_size=224) -> torch.Tensor:
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


def init_models(version: str, precision: str = "fp16", load_in_4bit: bool = False, 
               load_in_8bit: bool = False, model_type: str = "ori") -> Tuple[PreTrainedTokenizer, torch.nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(
        version,
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

    if model_type=="sam2":
        from model.evf_sam2_video import EvfSam2Model
        model = EvfSam2Model.from_pretrained(
            version, low_cpu_mem_usage=True, **kwargs
        )

    if (not load_in_4bit) and (not load_in_8bit):
        model = model.cuda()
    model.eval()

    return tokenizer, model


def main(version: str, vis_save_path: str = "./infer", precision: str = "fp16", image_size: int = 224, 
         load_in_8bit: bool = False, load_in_4bit: bool = False, model_type: str = "ori", 
         image_path: str = "assets/zebra.jpg", prompt: str = "zebra top left") -> None:
    # use float16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # clarify IO
    if not os.path.exists(image_path):
        print("File not found in {}".format(image_path))
        exit()

    os.makedirs(vis_save_path, exist_ok=True)

    # initialize model and tokenizer
    tokenizer, model = init_models(version, precision, load_in_4bit, load_in_8bit, model_type)

    # preprocess    
    image_np = cv2.imread(image_path+"/00000.jpg")
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # original_size_list = [image_np.shape[:2]]

    image_beit = beit3_preprocess(image_np, image_size).to(dtype=model.dtype, device=model.device)

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=model.device)

    # infer
    output = model.inference(
        image_path,
        image_beit.unsqueeze(0),
        input_ids,
        # original_size_list=original_size_list,
    )
    # save visualization
    files = os.listdir(image_path)
    files.sort()
    for i, file in enumerate(files):
        img = cv2.imread(os.path.join(image_path, file))
        out = img + np.array([0,0,128]) * output[i][1].transpose(1,2,0)
        cv2.imwrite(os.path.join(vis_save_path, file), out)


if __name__ == "__main__":
    # If command line arguments are provided, use them to override defaults
    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(description="EVF infer")
        parser.add_argument("--version", required=True)
        parser.add_argument("--vis_save_path", default="./infer", type=str)
        parser.add_argument(
            "--precision",
            default="fp16",
            type=str,
            choices=["fp32", "bf16", "fp16"],
            help="precision for inference",
        )
        parser.add_argument("--image_size", default=224, type=int, help="image size")
        parser.add_argument("--load_in_8bit", action="store_true", default=False)
        parser.add_argument("--load_in_4bit", action="store_true", default=False)
        parser.add_argument("--model_type", default="ori", choices=["ori", "effi", "sam2"])
        parser.add_argument("--image_path", type=str, default="assets/zebra.jpg")
        parser.add_argument("--prompt", type=str, default="zebra top left")
        
        args = parser.parse_args(sys.argv[1:])
        
        main(
            version=args.version,
            vis_save_path=args.vis_save_path,
            precision=args.precision,
            image_size=args.image_size,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            model_type=args.model_type,
            image_path=args.image_path,
            prompt=args.prompt
        )
    else:
        # Require at least the version parameter
        print("Error: --version parameter is required")
        exit(1)