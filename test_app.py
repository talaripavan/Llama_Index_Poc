import pytest
import os
import json
from typing import Dict

# Importing all the DeepEval Modules
from deepeval import assert_test
from deepeval.metrics import BiasMetric
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# Setting out the metrics 
bias = BiasMetric(threshold=0.5)
contextual_precision = ContextualPrecisionMetric(threshold=0.5)
contextual_recall = ContextualRecallMetric(threshold=0.5)
answer_relevancy = AnswerRelevancyMetric(threshold=0.5)
faithfulness = FaithfulnessMetric(threshold=0.5)

evaluation_metrics = [
    bias,
    answer_relevancy,
    '''
    Reason behind commenting --> the complier is expecting the --> expected_output which cannot be None in the given below metrics.
    To run this we need to even declare the expected output .
    contextual_precision,
    contextual_recall,
    faithfulness
    '''
]

# Importing all the modules related to Llama-Index
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import Settings

# loading the secrets
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def create_query_engine_from_documents(data_directory: str, collection_name: str, dim: int = 1536):
    """
        data_directory (str): The directory containing the documents.
    """
    documents = SimpleDirectoryReader(data_directory).load_data()
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    vector_store = MilvusVectorStore(dim=dim, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)   
    llm = OpenAI(model="gpt-3.5-turbo")   
    rag_application = index.as_query_engine(llm=llm)
    return rag_application
rag_application = create_query_engine_from_documents('data1', 'llamacollection')

# Testing purpose 
# response = rag_application; user_input = input("Ask a query"); response_object = response.query(user_input); print(response_object)

file_path = "DeepEval/Evaluation_data/test1.json"
with open(file_path, "r") as json_file:
    json_data = json.load(json_file)
 
   
@pytest.mark.parametrize(
    "input_output_pair",
    json_data,
)
def test_llamaindex(input_output_pair: Dict):
    input = input_output_pair.get("input", None)
    expected_output = input_output_pair.get("expected_output", None)
    response_object = rag_application.query(input)
    if response_object is not None:
        actual_output = response_object.response
        retrieval_context = [node.get_content() for node in response_object.source_nodes]
    actual_output = actual_output
    retrieval_context = retrieval_context
    print(retrieval_context)
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected_output
    )
    assert_test(test_case, evaluation_metrics)
