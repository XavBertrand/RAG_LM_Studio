from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
)
from transformers import AutoTokenizer
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import matplotlib.pyplot as plt
import tiktoken
import sys, os
import shutil
from utils_raptor import *



embedding_db = None

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]


## Helper Fuction to count the number of Tokensin each text
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def embed_raptor(
    config,
):
    data_directory = config["content_path"]
    embedding_directory = os.path.join(config["content_path"], "chroma_db_raptor")

    if os.path.exists(embedding_directory):
        shutil.rmtree(embedding_directory)

    # Load the huggingface embedding model
    model_name = config["embedding_model"]

    encode_kwargs = {
        "normalize_embeddings": True,  # set True to compute cosine similarity
    }

    model_kwargs = {
        "device": "cuda",
        # "device": "cpu",
    }

    # embedding_model = HuggingFaceEmbeddings(
    #     model_name=model_name,
    #     model_kwargs=model_kwargs,
    #     encode_kwargs=encode_kwargs,
    #     show_progress=True,
    # )
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda", "trust_remote_code": True},
        encode_kwargs=encode_kwargs,
        show_progress=True,

    )
    # embedding_model = OpenAIEmbeddings()

    print("\nCalculating Embeddings\n")

    # Load the text from the data directory
    # loader = DirectoryLoader(data_directory,
    #                          glob="*.txt",
    #                          loader_cls=TextLoader)

    loader = DirectoryLoader(data_directory, glob="**/*.txt")

    documents = loader.load()

    docs_texts = [d.page_content for d in documents]

    # compute the nb of token for each doc
    counts = [num_tokens_from_string(d.page_content, "cl100k_base") for d in documents]

    # Doc texts concat
    d_sorted = sorted(documents, key=lambda x: x.metadata["source"])
    d_reversed = list(reversed(d_sorted))
    concatenated_content = "\n\n\n --- \n\n\n".join(
        [doc.page_content for doc in d_reversed]
    )
    print(
        "Num tokens in all context: %s"
        % num_tokens_from_string(concatenated_content, "cl100k_base")
    )

    plot = False

    if plot:
        # Plotting the histogram of token counts
        plt.figure(figsize=(10, 6))
        plt.hist(counts, bins=30, color="blue", edgecolor="black", alpha=0.7)
        plt.title("Histogram of Token Counts")
        plt.xlabel("Token Count")
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.75)

        # Display the histogram
        plt.show()

    chunk_size_tok = config["chunk_size"]
    # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=chunk_size_tok,
    #     chunk_overlap=0,
    # )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )
    texts_split = text_splitter.split_text(concatenated_content)

    # Build tree
    leaf_texts = texts_split
    # results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3, embedding=embedding_model, llm_raptor=llm_raptor)
    results = recursive_embed_cluster_summarize(
        leaf_texts,
        level=1,
        n_levels=config["n_levels_raptor"],
        embedding=embedding_model,
        llm_raptor=config["llm_ollama"],
    )

    # Initialize all_texts with leaf_texts
    all_texts = leaf_texts.copy()

    # Iterate through the results to extract summaries from each level and add them to all_texts
    for level in sorted(results.keys()):
        # Extract summaries from the current level's DataFrame
        summaries = results[level][1]["summaries"].tolist()
        # Extend all_texts with the summaries from the current level
        all_texts.extend(summaries)

    # Now, use all_texts to build the vectorstore with Chroma
    # Vector store must be done of truncated texts!! Crash if nb tokens>512
    all_texts_truncated = limit_tokens(all_texts, max_tokens=500)
    embedding_db = Chroma.from_texts(
        texts=all_texts_truncated,
        embedding=embedding_model,
        persist_directory=embedding_directory,
    )

    print("Embeddings completed")
