import os
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex , get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.colbert_rerank import ColbertRerank

collection_name = os.getenv('COLLECTION_NAME')
vector_store = MilvusVectorStore(collection_name=collection_name, dim=1536, overwrite=False, uri="http://192.168.29.44:19530")
index = VectorStoreIndex.from_vector_store(vector_store)
retriever = VectorIndexRetriever(index=index,similarity_top_k=3)
response_synthesizer = get_response_synthesizer()
colbert_reranker = ColbertRerank(
    top_n=5,
    model="colbert-ir/colbertv2.0",
    tokenizer="colbert-ir/colbertv2.0",
    keep_retrieval_score=True,
)
retriever_rerank = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[colbert_reranker]
)
retriever_rerank_response = retriever_rerank.query(" What is the full name for the college code 'ACEG' ? ")

print(retriever_rerank_response,"\n")

for node in retriever_rerank_response.source_nodes:
    print(node.id_)
    print(node.node.get_content()[:120])
    print("reranking score: ", node.score)
    print("retrieval score: ", node.node.metadata["retrieval_score"])
    print("**********")