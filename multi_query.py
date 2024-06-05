from llama_index.core.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

collection_name = "llamacollection"

gpt35 = OpenAI(temperature=0, model="gpt-3.5-turbo")
step_decompose_transform_gpt3 = StepDecomposeQueryTransform(
    llm=gpt35, verbose=True
)
index_summary = "Used to answer questions about the author"

vector_store = MilvusVectorStore(collection_name=collection_name, dim=1536, overwrite=False, uri="http://192.168.29.44:19530")
index = VectorStoreIndex.from_vector_store(vector_store)

rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3
)

query_engine = index.as_query_engine(
    llm=gpt35,
    similarity_top_k=5, 
    node_postprocessors=[rerank]
    )

query_engine = MultiStepQueryEngine(
    query_engine=query_engine,
    query_transform=step_decompose_transform_gpt3,
    index_summary=index_summary,
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
     description=("Utilize this tool to efficiently retrieve documents based on specified criteria."),
    )

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