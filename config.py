class Settings:
    MODEL_ID = "qwen-plus"
    MODEL_KEY = "sk-223f07fbb08d4db99db4a2292bc9cf2d"
    MODEL_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    MODEL_SYS_PROMPT = """Translate this json into pure English and use natural language to describe.
                      The sentence formats should adhere to 'The xxx is xxx'. The overall result
                      should be concatenated to form a paragraph. Please be concise and do not
                      add adjective information and cling to the original semantics. Afterwards,
                      you should use comma to concatenate all the sentences."""

    DATA_DIR = "dataset"
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
    CHAT_TEMPLATE = "<image>\nBackground on this image: {prompt}\nConsidering {metric}, how would you rate this image?"


settings = Settings()
