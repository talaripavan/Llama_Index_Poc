from llama_index.core import SimpleDirectoryReader , VectorStoreIndex , StorageContext , get_response_synthesizer
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank

from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
from dotenv import load_dotenv
load_dotenv()

docs = SimpleDirectoryReader(input_files=["paul-graham-ideas.pdf"])
reader = docs.load_data()

node_parser = HierarchicalNodeParser.from_defaults()
nodes = node_parser.get_nodes_from_documents(reader)

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

leaf_nodes = get_leaf_nodes(nodes)

vector_store = MilvusVectorStore(collection_name="llamacollection",dim=1536,overwrite=True,uri="http://192.168.29.44:19530")
storage_context = StorageContext.from_defaults(docstore=docstore,vector_store=vector_store)

automerging_index = VectorStoreIndex(
    storage_context=storage_context,nodes = leaf_nodes
)

base_retriever = automerging_index.as_retriever(similarity_top_k=5)
retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)

rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3
)
response_synthesizer = get_response_synthesizer()
automerging_rerank = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[rerank]
)

response = automerging_rerank.query("Who was in the first batch of the accelerator program the author started?")
print(response)
