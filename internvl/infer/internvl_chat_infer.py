import os
import math
import json
import torch
import argparse
import torchvision.transforms as T

from pathlib import Path

from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from config import settings
from data.schema import AnnotationBody
from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.constants import IMAGENET_MEAN, IMAGENET_STD


class Evaluator:
    def __init__(
        self,
        image_path: str,
        text_path: str,
        model_path: str,
        model_name: str = "InternVL3-2B",
    ):
        self.image_path = Path(image_path)
        self.text_path = Path(text_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )

        self.model: InternVLChatModel = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=self.split_model(model_name),
        ).eval()

        self.generation_config = dict(max_new_tokens=1024, do_sample=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def build_transform(input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
        return transform

    def find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(
        self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
    ):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert("RGB")
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def split_model(self, model_name: str):
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers = {
            "InternVL3-1B": 24,
            "InternVL3-2B": 24,
            "InternVL3-4B": 36,
            "InternVL3-8B": 32,
            "InternVL3-26B": 48,
            "InternVL3-38B": 64,
            "InternVL3-78B": 80,
        }[model_name]
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f"language_model.model.layers.{layer_cnt}"] = i
                layer_cnt += 1
        device_map["vision_model"] = 0
        device_map["mlp1"] = 0
        device_map["language_model.model.tok_embeddings"] = 0
        device_map["language_model.model.embed_tokens"] = 0
        device_map["language_model.output"] = 0
        device_map["language_model.model.norm"] = 0
        device_map["language_model.lm_head"] = 0
        device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

        return device_map

    def evaluate_single(self, anno: AnnotationBody):
        ip = self.image_path / anno.image_id
        pixel_values = self.load_image(ip, max_num=12).to(torch.bfloat16)
        pixel_values = pixel_values.to(self.device)

        for metric in settings.METRICS.keys():
            question = settings.CHAT_TEMPLATE.format(
                metric=metric,
                prompt=json.dumps(anno.prompt),
            )

            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                self.generation_config,
            )

            print(f"Image Path: {ip}, From {metric}, {response}")

    @torch.no_grad()
    def evaluate(self):
        for p in os.listdir(self.text_path):
            tp = self.text_path / p

            with open(tp, "r+") as f:
                anno = AnnotationBody(**json.load(f))
                self.evaluate_single(anno)


parser = argparse.ArgumentParser()
parser.add_argument("--image_path", required=True)
parser.add_argument("--text_path", required=True)
parser.add_argument("--model_path", required=True)


args = parser.parse_args()
evl = Evaluator(args.image_path, args.text_path, args.model_path)
