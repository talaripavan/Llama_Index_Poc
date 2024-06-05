""" Complete this course for better understanding of sentence window retrieval . """
# https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/1/introduction

from llama_index.core import SimpleDirectoryReader , VectorStoreIndex , StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

docs = SimpleDirectoryReader(input_files=["paul-graham-ideas.pdf"])
reader = docs.load_data()
'''
print(type(reader), "\n")
print(len(reader), "\n")
print(type(reader[0]))
print(reader[0])
'''

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
window_nodes = node_parser.get_nodes_from_documents(documents=reader)

vector_store = MilvusVectorStore(collection_name="llamacollection",dim=1536,overwrite=False,uri="http://192.168.29.44:19530")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
window_vector_index = VectorStoreIndex(nodes=window_nodes,storage_context=storage_context)

query_engine = window_vector_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)

''' 
window_response = query_engine.query(
    " Summarize the author's journey in 5 lines "
)
print(window_response)

window = window_response.source_nodes[0].node.metadata["window"]
print(f"Window: {window}")

'''

vector_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
     description=("Utilize this tool to efficiently retrieve documents based on specified criteria."),
    )

gpt35 = OpenAI(temperature=0, model="gpt-3.5-turbo")

agent_worker = FunctionCallingAgentWorker.from_tools(
tools=[vector_tool],
llm=gpt35,
system_prompt="""
    You are an agent designed to answer queries over a set of given papers.
    Please always use the tools provided to answer a question.Do not rely on prior knowledge.""",
    verbose=True
    )
agent = AgentRunner(agent_worker)
response = agent.chat_repl()

