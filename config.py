import bisect


class Settings:
    DATA_DIR = "dataset"

    DATA_PRETRAIN_META = "pretrain_meta"
    DATA_PRETRAIN_ANNO = "pretrain_anno"
    DATA_PRETRAIN_NAME = "aigc-mos"

    DATA_ANNO = "annotation"
    DATA_META = "meta"
    DATA_NAME = "aigc-iqa"

    METRICS = {
        "Subject Consistency": "主体一致性",
        "Subject Quantity Consistency": "主体数量一致性",
        "Subject Attribute Consistency": "主体属性一致性",
        "Scene Completeness": "场所完整性",
        "Spatial Relationship Consistency": "空间关系一致性",
        "Style Compatibility": "风格契合度",
        "Logical Coherence": "逻辑合理性",
    }

    QUALITY_MAP = ["bad", "good"]
    RESPONSE_TEMPLATE = "The quality of this image is {quality}."
    CHAT_TEMPLATE = "<image>\nEvaluate this image for '{metric}' based on the following JSON (keys in Chinese but respond in English):\n\n```json\n{prompt}\n```.\nOutput format: 'The quality of this image is [good/bad].'"

    MOS_QUALITY_SCALE = [2.8, 3.4, 4.0, 4.45, 5]
    MOS_QUALITY_MAP = ["bad", "poor", "fair", "good", "excellent"]

    @classmethod
    def MOS_TO_SCALE(cls, mos: float):
        return cls.MOS_QUALITY_MAP[bisect.bisect_right(cls.MOS_QUALITY_SCALE, mos)]

    PRETRAIN_CHAT_TEMPLATE = "<image>\nEvaluate this image based on the following description: {prompt}. Output format: 'The quality of this image is [bad/poor/fair/good/excellent].'"

    LORA_WEIGHT = "lora_weights.pth"


settings = Settings()
