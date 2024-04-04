import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import Settings

# Importing all the modules related to Llama-Index
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import Settings

# loading the secrets
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def create_query_engine_from_documents(data_directory: str, collection_name: str, dim: int = 1536):
    documents = SimpleDirectoryReader(data_directory).load_data()
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    vector_store = MilvusVectorStore(dim=dim, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)   
    llm = OpenAI(model="gpt-3.5-turbo")   
    rag_application = index.as_query_engine(llm=llm)
    return rag_application

respone = create_query_engine_from_documents('data1', 'llamacollection')


def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    st.set_page_config(page_title="Chat with Multiple Pdfs",page_icon=":books:")
    st.header("Chat with muiltipple pdfs")
    st.text_input("Ask a question")
                         
if __name__ == '__main__':
    main()