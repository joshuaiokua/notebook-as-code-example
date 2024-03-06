"""
Defines a class that encapsulates the entire flow of creating a Retrieve-and-Generate (RAG) setup, including document loading, node parsing, service context creation, retriever initialization, and query engine setup.
File generated on: 2024-03-06 09:15:55
"""

from src.imports import *

class CreateRAG():
    def __init__(self):
        self.document = None
        self.base_nodes = None
        self.service_context = None
        self.base_retriever = None
        self.query_engine = None
    
    def load_document(self, file_source:str, file_name:str = "file.pdf"):
        # Retrieve the document
        command = f'wget --user-agent "Mozilla" "{file_source}" -O "{file_name}"'
        subprocess.run(command, check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Load the document
        loaded_document = PDFReader().load_data(file=Path(file_name))

        # Get document text and generate a Document object
        document_text = "\n\n".join([d.get_content() for d in loaded_document])
        self.document = [Document(text=document_text)]

    def parse_document_to_nodes(self, node_parser = SimpleNodeParser, chunk_size:int = 1024):
        # Parse the document into nodes
        node_parser = node_parser.from_defaults(chunk_size=chunk_size)

        # Set base nodes and set node ids to be a constant
        self.base_nodes = node_parser.get_nodes_from_documents(self.document)
        for idx, node in enumerate(self.base_nodes):
            node.id_ = f"node-{idx}"

    def create_service_context(self, llm_model = "gpt-3.5-turbo", embedding_model = "local:BAAI/bge-small-en"):
        load_dotenv() # Need to have .env file with OpenAI API key
        self.service_context = ServiceContext.from_defaults(
            llm=OpenAI(model=llm_model), 
            embed_model=resolve_embed_model(embedding_model)
        )

    def create_retriever(self, vector_store_impl = VectorStoreIndex, similarity_top_k = 2):
        self.base_retriever = vector_store_impl(self.base_nodes, service_context=self.service_context).as_retriever(similarity_top_k=similarity_top_k)

    def create_query_engine(self, query_engine = RetrieverQueryEngine):
        self.query_engine = query_engine.from_args(
            self.base_retriever, service_context=self.service_context
        )

    def deploy(self, file_source:str, file_name:str = "file.pdf",
               node_parser = SimpleNodeParser, chunk_size:int = 1024,
               llm_model = "gpt-3.5-turbo", embedding_model = "local:BAAI/bge-small-en",
               vector_store_impl = VectorStoreIndex, similarity_top_k = 2,
               query_engine = RetrieverQueryEngine):
        self.load_document(file_source, file_name)
        self.parse_document_to_nodes(node_parser, chunk_size)
        self.create_service_context(llm_model, embedding_model)
        self.create_retriever(vector_store_impl, similarity_top_k)
        self.create_query_engine(query_engine)
        return self

    def retrieve(self, query:str):
        if self.base_retriever is not None:
            retrievals = self.base_retriever.retrieve(query)
            return retrievals
        else:
            raise ValueError("Query engine not created")
            

    def query(self, query:str):
        if self.query_engine is not None:
            response = self.query_engine.query(query)
            return str(response)
        else:
            raise ValueError("Query engine not created")

