"""
Handles the loading and parsing of documents from a given source. This includes functionality for downloading documents and extracting text data to form document objects.
File generated on: 2024-03-06 08:47:23
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

