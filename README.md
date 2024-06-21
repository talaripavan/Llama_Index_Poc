# Retrieval-Augmented Generation (RAG) Project

This repository is all about RAG, a technique that's like having a super-powered search engine for your company's data. Imagine a personalized ChatGPT, but instead of chatting about anything, it helps you find and use the specific knowledge you need – fast!

## Introduction
This repository delves into the Retrieval-Augmented Generation (RAG), a cutting-edge technique designed to empower large language models (LLMs) with the power of information retrieval. By combining these two forces, RAG unlocks the potential for generating incredibly informative and contextually relevant text, making it a game-changer for applications like chatbots, customer service automation, and more.

## Repository Contents

- **chat_engine.py**: This Python script serves as a foundational blueprint for a complete RAG pipeline. 
  
- **chat_with_docs.py**: This interactive POC utilizes Streamlit to create a user-friendly interface. Experiment with document retrieval and generation in a dynamic setting!
  
- **colbert_re-ranker.py**: This script retrieves relevant documents based on a user query and then utilizes the mighty ColBERT for re-ranking. ColBERT meticulously prioritizes the most fitting documents before sending them to the LLM for text generation, ensuring the highest quality output.
  
- **flag_embedding_re-ranker.py**:Similar to the ColBERT script, this one leverages FLAG embeddings for re-ranking retrieved documents. By meticulously selecting the most relevant documents, FLAG embeddings ensure the LLM receives the best possible information to work with.

- **milvus_hybrid_search.py**: Efficiency is key! This script demonstrates a hybrid vector search approach for retrieving relevant documents when a user submits a query. This innovative approach optimizes the search process, ensuring you get the information you need quickly.

- **sentence_transformer_rerank.py**:Don't underestimate the power of fine-tuning! This script employs sentence transformers for re-ranking, meticulously selecting the most relevant documents before feeding them to the LLM. This approach ensures the LLM focuses its power on the most informative data.

- **multi_query.py** : This approach breaks down complex user queries into smaller, more manageable sub-queries. By doing this, the system can retrieve more relevant documents for each sub-query and then combine the results to answer the original user query. This can lead to more accurate and informative responses.

