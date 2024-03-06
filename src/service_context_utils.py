"""
Provides functionalities to create service contexts for language models and embedding models, including initializing service contexts with required models.
File generated on: 2024-03-06 08:55:03
"""

from src.imports import *

def create_service_context(llm_model = "gpt-3.5-turbo", embedding_model = "local:BAAI/bge-small-en"):
    load_dotenv() # Need to have .env file with OpenAI API key
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model=llm_model), 
        embed_model=resolve_embed_model(embedding_model)
    )
    return service_context

def create_retriever(base_nodes, service_context, similarity_top_k = 2, vector_store_impl = VectorStoreIndex):
    base_index = vector_store_impl(base_nodes, service_context=service_context)
    base_retriever = base_index.as_retriever(similarity_top_k=similarity_top_k)
    return base_retriever

def create_query_engine(retriever, service_context, query_engine = RetrieverQueryEngine):
    query_engine = query_engine.from_args(
        retriever, service_context=service_context
    )
    return query_engine

