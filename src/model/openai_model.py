import os
from dotenv import load_dotenv

MODEL = "gpt-4o"
MODEL_PROVIDER = "openai"

def get_openai_model():
    try:
        load_dotenv()
    except ImportError:
        pass

    from langchain.chat_models import init_chat_model

    return init_chat_model(MODEL, model_provider=MODEL_PROVIDER)