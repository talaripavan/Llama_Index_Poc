from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.storage import StorageContext
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer



load_dotenv()
documents_path = "./documents"

def initialize():
    documents = SimpleDirectoryReader(documents_path).load_data()
    vector_store = MilvusVectorStore(dim=1536, collection_name="llamacollection")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index

index = initialize()

llm = OpenAI(model="gpt-3.5-turbo")

# To Run the Context Chat Mode these are some things we need to specify
'''
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        " You are a chatbot, able to have normal interactions, as well as talk"
        " about financial's,money, investing, or saving. "
    ),
)
'''
#chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose = True)
#chat_engine = index.as_chat_engine(chat_mode="openai",verbose = True)
chat_engine = index.as_chat_engine(chat_mode="best" , verbose = True )
#chat_engine = index.as_chat_engine(chat_mode="react" , verbose = True )

#input = "Compare cloud services & license support with cloud license & on-premise license models ?"
#response = chat_engine.chat(input)
chat_engine.chat_repl()