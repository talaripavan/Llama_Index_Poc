from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.memory import (
    VectorMemory,
    SimpleComposableMemory,
    ChatMemoryBuffer,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
load_dotenv()

vector_memory = VectorMemory.from_defaults(
    vector_store=None,  # leave as None to use default in-memory vector store
    embed_model=OpenAIEmbedding(),
    retriever_kwargs={"similarity_top_k": 2},
)

chat_memory_buffer = ChatMemoryBuffer.from_defaults()

composable_memory = SimpleComposableMemory.from_defaults(
    primary_memory=chat_memory_buffer,
    secondary_memory_sources=[vector_memory],
)

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b

def mystery(a: int, b: int) -> int:
    """Mystery function on two numbers"""
    return a**2 - b**2

multiply_tool = FunctionTool.from_defaults(fn=multiply)
mystery_tool = FunctionTool.from_defaults(fn=mystery)

llm = OpenAI(model="gpt-3.5-turbo")
agent = FunctionCallingAgent.from_tools(
    [multiply_tool, mystery_tool],
    llm=llm,
    memory=composable_memory,
    verbose=True
)
response = agent.chat("What is the mystery function on 5 and 6?")
response_1 = agent.chat("What happens if you multiply 2 and 3?")
print(response_1)