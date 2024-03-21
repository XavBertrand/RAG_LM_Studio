from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys

# data_directory = "./content"
# embedding_directory = "./content/chroma_db"
data_directory = (
    r"C:\Users\bertr\PycharmProjects\TestRAG\simple-rag-lmstudio-main\content"
)
embedding_directory = (
    r"C:\Users\bertr\PycharmProjects\TestRAG\simple-rag-lmstudio-main\content\chroma_db"
)

embedding_db = None


def embed(embedding_model="dangvantuan/sentence-camembert-base"):
    print("\nCalculating Embeddings\n")

    # Load the text from the data directory
    # loader = DirectoryLoader(data_directory,
    #                          glob="*.txt",
    #                          loader_cls=TextLoader)

    loader = DirectoryLoader(data_directory, glob="**/*.txt")

    documents = loader.load()

    # Split the data into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)

    chunks = text_splitter.split_documents(documents)

    # Load the huggingface embedding model
    # model_name = "BAAI/bge-base-en-v1.5"
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # model_name = "dangvantuan/sentence-camembert-base"
    model_name = embedding_model

    encode_kwargs = {
        "normalize_embeddings": True
    }  # set True to compute cosine similarity

    # embedding_model = HuggingFaceBgeEmbeddings(
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        # model_kwargs={'device': 'cpu'},
        model_kwargs={"device": "cuda"},
        encode_kwargs=encode_kwargs,
        show_progress=True,
    )

    embedding_db = Chroma.from_documents(
        chunks, embedding_model, persist_directory=embedding_directory
    )

    print("Embeddings completed")
