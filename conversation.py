from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
import os

openai_api_key = os.environ.get("OPENAI_API_KEY")

class Conversation_RAG:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name

    def create_vectordb(self):
        vectordb = FAISS.load_local("./db/faiss_index", OpenAIEmbeddings())
        
        return vectordb
    
    def create_model(self, max_new_tokens=512, temperature=0.1):

        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=self.model_name,
            temperature=temperature,
            max_tokens=max_new_tokens,
            )
    
        return llm
    
    def create_conversation(self, model, vectordb, k_context=5, instruction="Use the following pieces of context to answer the question at the end by. Generate the answer based on the given context only. If you do not find any information related to the question in the given context, just say that you don't know, don't try to make up an answer. Keep your answer expressive."):
        
        print(instruction)

        template = instruction + """
        context:\n
        {context}\n
        data: {question}\n
        """

        QCA_PROMPT = PromptTemplate(input_variables=["instruction", "context", "question"], template=template)

        qa = ConversationalRetrievalChain.from_llm(
            llm=model,
            chain_type='stuff',
            retriever=vectordb.as_retriever(search_kwargs={"k": k_context}),
            combine_docs_chain_kwargs={"prompt": QCA_PROMPT},
            get_chat_history=lambda h: h,
            verbose=True
        )
        return qa