import json
import asyncio

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from config import settings


class Preprocessor:
    def __init__(self):
        self._llm = AsyncOpenAI(api_key=settings.MODEL_KEY, base_url=settings.MODEL_URL)

    async def _chat(self, user_input: str) -> str:
        res: ChatCompletion = await self._llm.chat.completions.create(
            model=settings.MODEL_ID,
            messages=[
                {"role": "system", "content": settings.MODEL_SYS_PROMPT},
                {"role": "user", "content": user_input},
            ],
        )

        return res.choices[0].message.content

    async def extract_caption_from_json(self, prompt: str):
        res = await self._chat(prompt)
        print(res)


prompt = {
    "生成目的": "游戏场景",
    "场所": "雪花纷飞的雪地上",
    "主体": [{"名称": "狼狗", "装饰": "紫绒披风", "数量": 4, "颜色": "黑褐色"}],
    "空间关系": "4只狼狗分为前后两排",
    "风格": "像素画",
}

"""
主体一致性 → Subject Consistency
场所完整性 → Scene Completeness
空间关系一致性 → Spatial Relationship Consistency
风格美合度 → Style Aesthetic Compatibility
逻辑合理性 → Logical Coherence
"""

p = Preprocessor()

asyncio.run(p.extract_caption_from_json(json.dumps(prompt)))
