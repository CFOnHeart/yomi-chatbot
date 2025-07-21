from typing import Optional, List

from src.database.chat_db import ChatDatabase
from src.database.faiss_document_db import FAISSDocumentDatabase
from src.tools.simple.math import add, multiply
from src.tools.simple.human_assistance import human_assistance

class SettingsStore:
    def __init__(self):
        self.chat_database : ChatDatabase = ChatDatabase("database/chat_history.db")
        self.document_database : FAISSDocumentDatabase = FAISSDocumentDatabase("database/documents.db")
        self.llm_model_name = "azure/gpt-4o"
        self.embedding_model_name = "azure/text-embedding-ada-002"
        self.tools = [add, multiply, human_assistance]
        self.retrival_document_detection_threshold = 0.7

default_setting_store = SettingsStore()