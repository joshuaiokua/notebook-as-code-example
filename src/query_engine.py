"""
Contains the core functionality for creating and managing query engines. This includes the creation of the query engine itself, along with methods for querying and retrieving information based on the processed nodes from the document.
File generated on: 2024-02-29 18:50:18
"""

from src.imports import *

def create_query_engine(retriever, service_context, query_engine = RetrieverQueryEngine):
    query_engine = query_engine.from_args(
        retriever, service_context=service_context
    )
    return query_engine

