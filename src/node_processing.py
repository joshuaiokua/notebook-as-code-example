"""
Facilitates the transformation of document content into nodes, as well as the parsing of documents into nodes using a specified node parser.
File generated on: 2024-03-06 09:15:55
"""

from src.imports import *

def parse_document_to_nodes(document, node_parser = SimpleNodeParser, chunk_size:int = 1024):
    # Parse the document into nodes
    node_parser = node_parser.from_defaults(chunk_size=chunk_size)

    # Set base nodes and set node ids to be a constant
    base_nodes = node_parser.get_nodes_from_documents(document)
    for idx, node in enumerate(base_nodes):
        node.id_ = f"node-{idx}"
    
    return base_nodes

