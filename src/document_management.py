"""
This file includes functions and methods for managing documents, including loading documents from a source, parsing documents into nodes, and functions for creating service contexts and retrievers related to document management and processing.
File generated on: 2024-02-29 18:50:18
"""

from src.imports import *

def load_document(source:str = "https://arxiv.org/pdf/2307.09288.pdf", file_name:str = "llama2.pdf"):

    # Retrieve the document
    command = f'wget --user-agent "Mozilla" "{source}" -O "{file_name}"'
    subprocess.run(command, check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Load the document
    loaded_document = PDFReader().load_data(file=Path("llama2.pdf"))

    # Get document text and generate a Document object
    document_text = "\n\n".join([d.get_content() for d in loaded_document])
    document = [Document(text=document_text)]

    return document

def parse_document_to_nodes(document, node_parser = SimpleNodeParser, chunk_size:int = 1024):
    # Parse the document into nodes
    node_parser = node_parser.from_defaults(chunk_size=chunk_size)

    # Set base nodes and set node ids to be a constant
    base_nodes = node_parser.get_nodes_from_documents(document)
    for idx, node in enumerate(base_nodes):
        node.id_ = f"node-{idx}"
    
    return base_nodes

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

