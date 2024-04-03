'''
This python file just includes about the datasets , What is Synthesizer ? What is EvaluationDataSet() ? 
What is Golden ?
'''

from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset

'''

What is Synthsizer ?
The Synthesizer class is a tool for creating synthetic data. It starts by using a Language Learning Model (LLM) to generate basic inputs. Then, it enhances these inputs to make them more realistic. The improved inputs are used to produce a list called "goldens," which becomes your EvaluationDataset.

'''
'''
# Use synthesizer directly
synthesizer = Synthesizer()
synthesizer.generate_goldens_from_docs(
    document_paths=['data1/Sec Filling.pdf'],
    max_goldens_per_document=5
)

synthesizer.save_as(
    file_type='json',
    directory='DeepEval/synthetic_data'  # Specify the directory here
)

'''

dataset = EvaluationDataset() 
synthesizer = Synthesizer()
dataset.generate_goldens_from_docs(   
    synthesizer=synthesizer,
    document_paths=['data1/Sec Filling.pdf'],
    max_goldens_per_document=10
)
dataset.save_as(
    file_type='json',
    directory='DeepEval/Evaluation_data'  
)
