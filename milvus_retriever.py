""" This is for retriving the similar nodes[Relavant Nodes according to the query] from Milvus Vector Database. """

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.vector_stores import VectorStoreQuery

import os
from dotenv import load_dotenv
load_dotenv()
uri = os.getenv('URI')

query_str = "Explain about Startups according to paul?"
embed_model = OpenAIEmbedding()
query_embedding = embed_model.get_query_embedding(query_str)
milvus_vector_store = MilvusVectorStore(collection_name="test_longrag",uri=uri, dim=1536, overwrite=False)
query_mode = "default"
vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=5, mode=query_mode
)

query_result = milvus_vector_store.query(vector_store_query)
print(query_result.similarities[0])