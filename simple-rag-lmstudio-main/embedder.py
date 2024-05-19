from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


embedding_db = None


def embed(embedding_model="dangvantuan/sentence-camembert-base", content_path=None):
    data_directory = content_path
    embedding_directory = os.path.join(content_path, "chroma_db")

    # Load the huggingface embedding model
    model_name = embedding_model

    encode_kwargs = {
        "normalize_embeddings": True
    }  # set True to compute cosine similarity

    # embedding_model = HuggingFaceBgeEmbeddings(
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},
        # model_kwargs={"device": "cpu"},
        encode_kwargs=encode_kwargs,
        show_progress=True,
    )

    print("\nCalculating Embeddings\n")

    # Load the text from the data directory
    loader = DirectoryLoader(data_directory, glob="**/*.txt")

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Number of text splits generated: {len(chunks)}")

    embedding_db = Chroma.from_documents(
        chunks, embedding_model, persist_directory=embedding_directory
    )

    print("Embeddings completed")
