import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.storage import StorageContext
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.vector_stores.milvus import MilvusVectorStore


load_dotenv()
# Path of our documents in our system
documents_path = "./documents"

Settings.llm = OpenAI(model="gpt-3.5-turbo")
@st.cache_resource(show_spinner=False)
def initialize():
    documents = SimpleDirectoryReader(documents_path).load_data()
    vector_store = MilvusVectorStore(dim=1536, collection_name="llamacollection")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index

index = initialize()

st.title(" Come On ! Ask me .")
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role" : "assistant" , "content" : "Ask me a question !"}
    ]

chat_engine = index.as_query_engine()

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking.."):
            response = chat_engine.query(prompt)
            st.write(response.response)
            pprint_response(response,show_source=True)
            message = {"role":"assistance","content": response.response}
            st.session_state.messages.append(message)