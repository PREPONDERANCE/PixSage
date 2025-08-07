import json
import random
import asyncio
import aiofiles
import argparse
import aiofiles.os as aos

from pathlib import Path
from typing import Dict, List, Union, Any

from PIL import Image
from tqdm.asyncio import tqdm

from config import settings
from .schema import (
    AnnotationBody,
    ChatInternVL,
    AnnotationInternVL,
    AnnotationMeta,
    AnnotationPretrain,
)


class PreTrainPreprocessor:
    def __init__(self, img_path: str, text_path: str, train_split: float):
        self._img_path = Path(img_path)
        self._text_path = Path(text_path)
        self._train_split = train_split

    def construct_data(self, name: str, detail: AnnotationPretrain):
        ip = self._img_path / name
        ip_relative = str(ip.relative_to(self._img_path))

        img = Image.open(ip).convert("RGB")
        w, h = img.width, img.height

        conv = [
            ChatInternVL(
                source="human",
                value=settings.PRETRAIN_CHAT_TEMPLATE.format(prompt=detail.prompt),
            ),
            ChatInternVL(
                source="gpt",
                value=settings.RESPONSE_TEMPLATE.format(
                    quality=settings.MOS_TO_SCALE(detail.mos)
                ),
            ),
        ]
        return AnnotationInternVL(
            id=ip_relative,
            image=ip_relative,
            width=w,
            height=h,
            score=detail.mos / 5,
            metric="overall",
            conversations=conv,
        )

    async def construct_file(self, annotations: List[AnnotationInternVL]):
        random.shuffle(annotations)

        total = len(annotations)
        train_size = int(total * self._train_split)

        await aos.makedirs(settings.DATA_DIR, exist_ok=True)

        path = Path(settings.DATA_DIR)
        anno_train_file = path / f"{settings.DATA_PRETRAIN_ANNO}_train.jsonl"
        anno_test_file = path / f"{settings.DATA_PRETRAIN_ANNO}_eval.jsonl"
        meta_file = path / f"{settings.DATA_PRETRAIN_META}.json"

        async with aiofiles.open(anno_train_file, "w+") as f:
            for i in range(train_size):
                a = annotations[i].model_dump(by_alias=True)
                await f.write(json.dumps(a, ensure_ascii=False) + "\n")

        async with aiofiles.open(anno_test_file, "w+") as f:
            for i in range(train_size, total):
                a = annotations[i].model_dump(by_alias=True)
                await f.write(json.dumps(a, ensure_ascii=False) + "\n")

        meta = AnnotationMeta(
            root=str(self._img_path.absolute()),
            annotation_train=str(anno_train_file.absolute()),
            annotation_eval=str(anno_test_file.absolute()),
            length=total,
        )
        async with aiofiles.open(meta_file, "w+") as f:
            await f.write(
                json.dumps(
                    {settings.DATA_PRETRAIN_NAME: meta.model_dump()},
                    ensure_ascii=False,
                )
            )

    async def convert(self):
        annotations = []

        async with aiofiles.open(self._text_path, "r+") as f:
            anno: Dict[str, Any] = json.loads(await f.read())

        for name, detail in tqdm(anno.items()):
            ap = AnnotationPretrain(**detail)
            annotations.append(await asyncio.to_thread(self.construct_data, name, ap))

        await self.construct_file(annotations)


class DataPreprocessor:
    def __init__(self, img_path: str, text_path: str, train_split: float):
        self._img_path = Path(img_path)
        self._text_path = Path(text_path)
        self._train_split = train_split

    async def get_prompt(self, prompt: Dict[str, Union[str, Dict[str, str]]]) -> str:
        return await asyncio.to_thread(json.dumps, prompt, ensure_ascii=False)

    def construct_data(
        self, anno: AnnotationBody, prompt: str
    ) -> List[AnnotationInternVL]:
        ip = self._img_path / anno.image_id
        ip_relative = str(ip.relative_to(self._img_path))

        img = Image.open(ip).convert("RGB")
        w, h = img.width, img.height

        annotations = []
        for metric, metric_zh in settings.METRICS.items():
            chats = [
                ChatInternVL(
                    source="human",
                    value=settings.CHAT_TEMPLATE.format(prompt=prompt, metric=metric),
                ),
                ChatInternVL(
                    source="gpt",
                    value=settings.RESPONSE_TEMPLATE.format(
                        quality=settings.QUALITY_MAP[anno.scores[metric_zh]]
                    ),
                ),
            ]

            annotations.append(
                AnnotationInternVL(
                    width=w,
                    height=h,
                    id=ip_relative,
                    image=ip_relative,
                    score=anno.scores[metric_zh],
                    metric=metric,
                    conversations=chats,
                )
            )

        return annotations

    async def construct_file(self, anno: List[AnnotationInternVL]):
        metric_size = len(settings.METRICS)
        train_size = int(len(anno) // metric_size * self._train_split) * metric_size
        test_size = len(anno) - train_size

        train_data, test_data = anno[:train_size], anno[train_size : len(anno)]
        random.shuffle(train_data)

        await aos.makedirs(settings.DATA_DIR, exist_ok=True)

        path = Path(settings.DATA_DIR)
        anno_train_file = path / f"{settings.DATA_ANNO}_train.jsonl"
        anno_test_file = path / f"{settings.DATA_ANNO}_eval.jsonl"
        meta_file = path / f"{settings.DATA_META}.json"

        async with aiofiles.open(anno_train_file, "w+") as f:
            for i in range(train_size):
                a = train_data[i].model_dump(by_alias=True)
                await f.write(json.dumps(a, ensure_ascii=False) + "\n")

        async with aiofiles.open(anno_test_file, "w+") as f:
            for i in range(test_size):
                a = test_data[i].model_dump(by_alias=True)
                await f.write(json.dumps(a, ensure_ascii=False) + "\n")

        meta = AnnotationMeta(
            root=str(self._img_path.absolute()),
            annotation_train=str(anno_train_file.absolute()),
            annotation_eval=str(anno_test_file.absolute()),
            length=len(anno) * len(settings.METRICS),
        )
        async with aiofiles.open(meta_file, "w+") as f:
            await f.write(
                json.dumps(
                    {settings.DATA_NAME: meta.model_dump()},
                    ensure_ascii=False,
                )
            )

    async def convert(self):
        annotations = []

        for p in tqdm(await aos.listdir(self._text_path)):
            if p == ".DS_Store":
                continue

            tp = self._text_path / p

            async with aiofiles.open(tp, "r+") as f:
                content = AnnotationBody(**json.loads(await f.read()))
                prompt = await self.get_prompt(content.prompt)
                annotations.extend(
                    await asyncio.to_thread(self.construct_data, content, prompt)
                )

        await self.construct_file(annotations)


parser = argparse.ArgumentParser(description="Preprocess image and text annotations")
parser.add_argument("--image_path", required=True, help="Path to the images directory")
parser.add_argument(
    "--text_path", required=True, help="Path to the text annotations directory"
)
parser.add_argument(
    "--train_split", type=float, default=0.8, help="Training and test dataset split"
)
parser.add_argument(
    "--type", type=str, default="finetune", help="Pretrain or finetune preprocessor"
)

args = parser.parse_args()

pc = DataPreprocessor if args.type == "finetune" else PreTrainPreprocessor
p = pc(args.image_path, args.text_path, args.train_split)
asyncio.run(p.convert())
