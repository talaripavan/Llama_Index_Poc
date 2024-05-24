import pytest
import os
import json
from typing import Dict
from deepeval import assert_test
'''
from deepeval.metrics.ragas import (
    RAGASContextualPrecisionMetric,
    RAGASFaithfulnessMetric,
    RAGASContextualRecallMetric,
    RAGASAnswerRelevancyMetric,
)
'''
from deepeval.metrics import BiasMetric
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate


from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import Settings

bias = BiasMetric(threshold=0.5)
#contextual_precision = ContextualPrecisionMetric(threshold=0.5)
#contextual_recall = ContextualRecallMetric(threshold=0.5)
answer_relevancy = AnswerRelevancyMetric(threshold=0.5)
#faithfulness = FaithfulnessMetric(threshold=0.5)


evaluation_metrics = [
  bias,
  #contextual_precision,
  #contextual_recall,
  answer_relevancy,
  #faithfulness
]

from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

documents = SimpleDirectoryReader('data1').load_data()

Settings.chunk_size = 512
Settings.chunk_overlap = 50

vector_store = MilvusVectorStore(dim=1536, collection_name="quick_setup")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Set up language model
llm = OpenAI(model="gpt-3.5-turbo")
    
# Create query engine
rag_application = index.as_query_engine(llm=llm)

# Open the file and load the JSON data
file_path = "DeepEval/Evaluation_data/test10.json"

# Open the file and load the JSON data
with open(file_path, "r") as json_file:
    json_data = json.load(json_file)
    
def generate_input_output_context_pairs_from_json(json_data: dict):
    in_out_context_pairs = []
    for entry in json_data:
        input_value = entry.get("input")
        context_value = entry.get("expected_output")
        if input_value:
            in_out_context_pairs.append({
                "input": input_value,
                "expected_output": entry.get("expected_output")
            })
    return in_out_context_pairs
        
input_output_pairs = generate_input_output_context_pairs_from_json(json_data)


@pytest.mark.parametrize(
    "input_output_pair",
    input_output_pairs,
)
def test_llamaindex(input_output_pair: Dict):
    input = input_output_pair.get("input", None)
    expected_output = input_output_pair.get("expected_output", None)

    # LlamaIndex returns a response object that contains
    # both the output string and retrieved nodes
    response_object = rag_application.query(input)

    # Process the response object to get the output string
    # and retrieved nodes
    if response_object is not None:
        actual_output = response_object.response
        #retrieval_context = [node.get_content() for node in response_object.source_nodes]

    actual_output = actual_output
    #retrieval_context = rag_application.get_retrieval_context() #add response here..

    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        #retrieval_context=retrieval_context,
        expected_output=expected_output
    )
    
    assert_test(test_case, evaluation_metrics)
    #print((test_case, evaluation_metrics))
    #evaluate([test_case], [evaluation_metrics])
