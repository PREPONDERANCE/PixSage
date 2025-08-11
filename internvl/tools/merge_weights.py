import torch
import argparse

from pathlib import Path
from transformers import AutoTokenizer

from config import settings
from internvl.model.internvl_chat import InternVLChatModel


def merge(model_path: str, dest_path: str):
    print("Loading model...")
    model = InternVLChatModel.from_pretrained(
        model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    ).eval()
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    lora_weight = torch.load(Path(model_path) / settings.LORA_WEIGHT, weights_only=True)
    model.load_state_dict(lora_weight, strict=False)

    if model.config.use_backbone_lora:
        model.vision_model.merge_and_unload()
        model.vision_model = model.vision_model.model
        model.config.use_backbone_lora = 0
    if model.config.use_llm_lora:
        model.language_model.merge_and_unload()
        model.language_model = model.language_model.model
        model.config.use_llm_lora = 0

    print("Saving model...")
    model.save_pretrained(dest_path)
    print("Saving tokenizer...")
    tokenizer.save_pretrained(dest_path)
    print("Done!")


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)

args = parser.parse_args()

model_path = Path(args.model_path)
merge(model_path, model_path / "model")
