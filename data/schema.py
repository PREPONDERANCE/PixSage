from typing import Dict, List, Any

from pydantic import BaseModel, Field


class AnnotationBody(BaseModel):
    image_id: str
    scores: Dict[str, int]
    prompt: Dict[str, Any]


class ChatInternVL(BaseModel):
    source: str = Field(alias="from")
    value: str

    class Config:
        populate_by_name = True


class AnnotationInternVL(BaseModel):
    id: str
    image: str
    width: int
    height: int
    score: int = 0
    metric: str = ""
    conversations: List[ChatInternVL]


class AnnotationMeta(BaseModel):
    root: str
    annotation_train: str
    annotation_eval: str
    data_augment: bool = False
    repeat_time: int = 1
    length: int
