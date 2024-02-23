"""
Contains the logic for initializing and managing the retrieval system. This includes creating the retriever based on vector similarity, setting up a query engine for processing and executing queries, and related setups for document search and information retrieval functionalities.
File generated on: 2024-02-23 06:12:30
"""

from src.imports import *

def create_retriever(base_nodes, service_context, similarity_top_k = 2, vector_store_impl = VectorStoreIndex):
    base_index = vector_store_impl(base_nodes, service_context=service_context)
    base_retriever = base_index.as_retriever(similarity_top_k=similarity_top_k)
    return base_retriever

def create_query_engine(retriever, service_context, query_engine = RetrieverQueryEngine):
    query_engine = query_engine.from_args(
        retriever, service_context=service_context
    )
    return query_engine

