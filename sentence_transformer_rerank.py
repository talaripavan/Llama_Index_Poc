from llama_index.core.postprocessor import SentenceTransformerRerank
import os
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex , get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

collection_name = os.getenv('COLLECTION_NAME')
vector_store = MilvusVectorStore(collection_name=collection_name, dim=1536, overwrite=False, uri="http")
index = VectorStoreIndex.from_vector_store(vector_store)
retriever = VectorIndexRetriever(index=index,similarity_top_k=3)
response_synthesizer = get_response_synthesizer()
rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3
)

retriever_rerank = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[rerank]
)

retriever_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
    )

retriever_rerank_response = retriever_rerank.query(" What is the full name for the college code 'ACEG' ? ")
retriever_engine_response = retriever_engine.query("What is the full name for the college code 'ACEG' ? ")

print("Retriever Response without Re-Rank",retriever_engine_response.get_formatted_sources(length=200),"\n","\n")
print("Retriever Response with Re-Rank",retriever_rerank_response.get_formatted_sources(length=200))

print(retriever_rerank_response)
