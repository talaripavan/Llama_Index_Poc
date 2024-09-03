from llama_index.core import SimpleDirectoryReader , VectorStoreIndex , StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.packs.longrag.base import split_doc , get_grouped_docs
from llama_index.llms.openai import OpenAI

import os
from dotenv import load_dotenv
load_dotenv()
uri = os.getenv('URI')


file = ["paul-graham-ideas.pdf"]
load_file = SimpleDirectoryReader(input_files=file).load_data()
chunk_size = 512
split_docs = split_doc(chunk_size=chunk_size,documents=load_file)
group_nodes = get_grouped_docs(nodes=split_docs)
splitter = SentenceSplitter()
sentence_splitter_nodes = splitter.get_nodes_from_documents(documents=load_file)

# It shows the difference between the Normal Sentence splitter nodes vs the group nodes by LongRAG.
#print("------------- Group Nodes ------------\n",group_nodes[1].get_content())
#print("------------- Normal Nodes ------------\n",sentence_splitter_nodes[1].get_content())

vector_store = MilvusVectorStore(collection_name="test_longrag",uri=uri, dim=1536, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
#sentence_index = VectorStoreIndex(nodes=sentence_splitter_nodes,storage_context=storage_context)
group_index = VectorStoreIndex(nodes=group_nodes,storage_context=storage_context)

# Vectorstore Index [Milvus]
vector_store = MilvusVectorStore(collection_name="test_longrag",uri=uri, dim=1536, overwrite=False)
vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = vector_index.as_query_engine(
    llm=OpenAI(model="gpt-4o-mini")
)
response = query_engine.query("Explain about Startups according to paul")
print(response)