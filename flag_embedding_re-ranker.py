import os
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex , get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

collection_name = os.getenv('COLLECTION_NAME')
vector_store = MilvusVectorStore(collection_name=collection_name, dim=1536, overwrite=False, uri="http")
index = VectorStoreIndex.from_vector_store(vector_store)
retriever = VectorIndexRetriever(index=index,similarity_top_k=3)
response_synthesizer = get_response_synthesizer()
flag_rerank = FlagEmbeddingReranker(model="BAAI/bge-reranker-large", top_n=5)

retriever_rerank = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[flag_rerank]
)

retriever_rerank_response = retriever_rerank.query(" What is the full name for the college code 'ACEG' ? ")
print(retriever_rerank_response)
print("Retriever Response with Re-Rank",retriever_rerank_response.get_formatted_sources(length=200))

# It takes a lot of time to run .
