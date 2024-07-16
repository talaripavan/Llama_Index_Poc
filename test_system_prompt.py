# I think system prompt plays a major role in the LLM response , as we observe the function output is same but the LLM response is changing when we are giving the system prompt .
""" 
system_prompt=
        You are an advanced RAG (Retrieval-Augmented Generation) agent. Your primary function is to process and analyze information from either PDF documents or Confluence URLs that I provide. Based on this information, you will answer my questions accurately and comprehensively. Your responses should be:
            1. Directly relevant to the content of the provided document or URL
            2. Precise and fact-based, avoiding speculation beyond the given information
            3. Structured in a clear, easy-to-understand format
            4. Capable of citing specific sections or pages when referring to the source material

            If you encounter any ambiguities or if the answer cannot be found in the provided content, please state this clearly. Always maintain the context of the document or URL in your responses. I am creating this RAG app, so please provide your responses in a way that demonstrates the effectiveness of the system. 
"""
from utils import milvus_vector_store , multi_query_transformation
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.llms.openai import OpenAI
from prompt_templates import step_back_prompt , few_shot_prompt , cot_prompt
import os
def response_agent(collection_name,prompt_template):
    vector_index = milvus_vector_store(collection_name=collection_name)
    vector_tool = multi_query_transformation(index=vector_index,response_synthesizer=prompt_template)
    llm = OpenAI(model="gpt-3.5-turbo")
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools=[vector_tool],
        llm=llm,
        verbose=True,
        )
    retrieve_agent = AgentRunner(agent_worker)
    return retrieve_agent

collection_name = os.getenv('COLLECTION_NAME')
agent = response_agent(collection_name=collection_name,prompt_template=step_back_prompt())
response = agent.chat("Explain about Paul's perspective about startups")
