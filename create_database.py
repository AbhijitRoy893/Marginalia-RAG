# create_database.py
# Load PDF(s) -> Split into chunks -> Create embeddings -> Store in Chroma
#
# This module can be run standalone (edit PDF_PATH below) or imported by
# app.py, which calls build_vectorstore() directly with uploaded files.

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():
    """Load the Hugging Face embedding model (shared by ingestion + querying)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_and_split(pdf_paths, chunk_size=1000, chunk_overlap=200):
    """
    Load one or more PDFs and split them into chunks.

    pdf_paths: str or list[str] - path(s) to PDF file(s)
    Returns: (chunks, num_pages)
    """
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(all_docs)
    return chunks, len(all_docs)


def build_vectorstore(pdf_paths, persist_directory="chroma_db", chunk_size=1000, chunk_overlap=200):
    """
    Full ingestion pipeline: load -> split -> embed -> store in Chroma.

    Returns: (vectorstore, num_pages, num_chunks)
    """
    chunks, num_pages = load_and_split(pdf_paths, chunk_size, chunk_overlap)
    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )
    return vectorstore, num_pages, len(chunks)


if __name__ == "__main__":
    # Standalone usage (adjust path as needed)
    PDF_PATH = "documents loaders/deeplearning.pdf"

    vectorstore, num_pages, num_chunks = build_vectorstore(PDF_PATH)

    print(f"Loaded {num_pages} pages")
    print(f"Created {num_chunks} chunks")
    print("PDF embeddings stored successfully in ChromaDB!")