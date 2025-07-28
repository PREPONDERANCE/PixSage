class Settings:
    MODEL_ID = "qwen-plus"
    MODEL_KEY = "sk-223f07fbb08d4db99db4a2292bc9cf2d"
    MODEL_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    MODEL_SYS_PROMPT = """Translate this into English and use natural language to describe.
                      The sentence formats should adhere to 'The xxx is xxx'. The overall result
                      should be concatenated to form a paragraph. Please be concise and do not
                      add adjective information and cling to the original semantics."""


settings = Settings()
