from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

file = ["./paul-graham-ideas.pdf"]
docs = SimpleDirectoryReader(input_files=file).load_data()
splitter = SentenceSplitter()
chunks = splitter.get_nodes_from_documents(documents=docs)
for node in chunks:
    hash_id = node.hash
    print(hash_id)