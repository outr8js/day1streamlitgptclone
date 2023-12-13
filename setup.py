from conversation import Conversation_RAG
from vector_index import *

class ModelSetup:
    def __init__(self, model_name):

        self.model_name = model_name

    def setup(self):

        conv_rag = Conversation_RAG(self.model_name)

        self.vectordb = conv_rag.create_vectordb()
        self.pipeline = conv_rag.create_model()

        return "Model Setup Complete"