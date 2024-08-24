from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


embedding_db = None


def embed(config):
    data_directory = config["content_path"]
    embedding_directory = os.path.join(config["content_path"], "chroma_db")

    # Load the huggingface embedding model
    model_name = config["embedding_model"]
    encode_kwargs = {"normalize_embeddings": True}

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda", "trust_remote_code": True},
        encode_kwargs=encode_kwargs,
        show_progress=True,

    )

    print("\nCalculating Embeddings\n")

    # Load the text from the data directory
    loader = DirectoryLoader(data_directory, glob="**/*.txt")

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Number of text splits generated: {len(chunks)}")

    embedding_db = Chroma.from_documents(
        chunks, embedding_model, persist_directory=embedding_directory
    )

    print("Embeddings completed")
