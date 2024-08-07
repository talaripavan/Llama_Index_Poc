from llama_index.core import SimpleDirectoryReader , VectorStoreIndex
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import (
    SemanticDoubleMergingSplitterNodeParser,
    LanguageConfig,
)

from dotenv import load_dotenv
load_dotenv()
# Mention your document path to load .
documents = SimpleDirectoryReader(input_files=["March-12(Quaterlly)-24.pdf"]).load_data()
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)

sentence_splitter = SentenceSplitter(chunk_size=512)
"""
Why I am using Gpt 4o mini model ?
GPT-4o mini is our most cost-efficient small model that's smarter and cheaper than GPT-3.5 Turbo, and has vision capabilities. The model has 128K context and an October 2023 knowledge cutoff.
Reference :- https://openai.com/api/pricing/
"""
llm = OpenAI(model="gpt-4o-mini")
semantic_nodes = semantic_splitter.get_nodes_from_documents(documents=documents)
sentence_nodes = sentence_splitter.get_nodes_from_documents(documents=documents)
config = LanguageConfig(language="english", spacy_model="en_core_web_md")
semantic_double_splitter = SemanticDoubleMergingSplitterNodeParser(
    language_config=config,
    initial_threshold=0.4,
    appending_threshold=0.5,
    merging_threshold=0.5,
    max_chunk_size=5000,
)
semantic_double_nodes = semantic_double_splitter.get_nodes_from_documents(documents)

print(" -------------  Semantic Nodes  ------------------- ")
print(semantic_nodes[24].get_content())

print(" -------------  Sentence Nodes  ------------------- ")
print(sentence_nodes[24].get_content())

print(" -------------  Semantic Double Nodes  ------------------- ")
print(semantic_double_nodes[24].get_content())


semantic_vector_index = VectorStoreIndex(nodes=semantic_nodes)
sentence_vector_index = VectorStoreIndex(nodes=sentence_nodes)
semantic_double_index = VectorStoreIndex(nodes=semantic_double_nodes)


semantic_query = semantic_vector_index.as_query_engine(llm=llm)
# Ask the question .
semantic_response = semantic_query.query(" ")
print(" -------------  Semantic Response  ------------------- ")
print(semantic_response)

sentence_query = sentence_vector_index.as_query_engine(llm=llm)
sentence_response = sentence_query.query("")
print(" -------------  Sentence Response  ------------------- ")
print(sentence_response)

semantic_double_query = semantic_double_index.as_query_engine(llm=llm)
semantic_double_response = sentence_query.query("")
print(" -------------  Semantic Double Response  ------------------- ")
print(semantic_double_response)
