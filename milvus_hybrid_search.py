from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex,StorageContext ,get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import os

collection_name = os.getenv('COLLECTION_NAME')

vector_store = MilvusVectorStore(
    collection_name=collection_name, 
    dim=1536, 
    overwrite=False, 
    uri="http",
    enable_sparse=True,
    hybrid_ranker="RRFRanker",
    hybrid_ranker_params={"k": 60},
    )
index = VectorStoreIndex.from_vector_store(vector_store)
retriever = VectorIndexRetriever(index=index,similarity_top_k=5,vector_store_query_mode="hybrid")
response_synthesizer = get_response_synthesizer()
retriever_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
    )
response = retriever_engine.query("What did the author learn?")
print(response)

