import json
import random
import asyncio
import aiofiles
import argparse
import aiofiles.os as aos

from pathlib import Path
from typing import Dict, List, Union

from PIL import Image
from tqdm.asyncio import tqdm

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from config import settings
from .schema import AnnotationBody, ChatInternVL, AnnotationInternVL, AnnotationMeta


class Preprocessor:
    def __init__(self, img_path: str, text_path: str, train_split: float):
        self._llm = AsyncOpenAI(api_key=settings.MODEL_KEY, base_url=settings.MODEL_URL)

        self._img_path = Path(img_path)
        self._text_path = Path(text_path)
        self._train_split = train_split

    async def _chat(self, user_input: str) -> str:
        res: ChatCompletion = await self._llm.chat.completions.create(
            model=settings.MODEL_ID,
            messages=[
                {"role": "system", "content": settings.MODEL_SYS_PROMPT},
                {"role": "user", "content": user_input},
            ],
        )

        return res.choices[0].message.content

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
        random.shuffle(anno)

        train_size = int(len(anno) * self._train_split)
        test_size = len(anno) - train_size

        await aos.makedirs(settings.DATA_DIR, exist_ok=True)

        path = Path(settings.DATA_DIR)
        anno_train_file = path / f"{settings.DATA_ANNO}_train.jsonl"
        anno_test_file = path / f"{settings.DATA_ANNO}_eval.jsonl"
        meta_file = path / f"{settings.DATA_META}.json"

        async with aiofiles.open(anno_train_file, "w+") as f:
            for i in range(train_size):
                a = anno[i].model_dump(by_alias=True)
                await f.write(json.dumps(a, ensure_ascii=False) + "\n")

        async with aiofiles.open(anno_test_file, "w+") as f:
            for i in range(train_size, train_size + test_size):
                a = anno[i].model_dump(by_alias=True)
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

args = parser.parse_args()

p = Preprocessor(args.image_path, args.text_path, args.train_split)
asyncio.run(p.convert())
