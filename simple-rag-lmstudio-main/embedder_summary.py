from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import os
from tqdm import tqdm
import shutil

embedding_db = None


def generate_summary(document, llm):
    prompt_template = """Résume le texte suivant en français. Le résumé doit être concis et contenir les informations les plus importantes du texte. Précise, les liens entre les personnes qui apparaissent dans le texte, et tu portes une attention particulière aux dates que tu fais apparaitre dans le résumé. Indique bien les prénoms et noms des personnes qui apparaissent dans le texte :

{text}

Résumé en français :"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

    summary = chain.run([Document(page_content=document.page_content)])
    return summary.strip()


def embed_summary(
    config
):
    data_directory = config["content_path"]
    embedding_directory = os.path.join(config["content_path"], "chroma_db_summary")

    if os.path.exists(embedding_directory):
        shutil.rmtree(embedding_directory)

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

    # Initialize Ollama LLM
    llm = Ollama(model=config["llm_ollama"])

    # Generate summaries and create new chunks
    summarized_chunks = []
    for doc in tqdm(documents):
        summary = generate_summary(doc, llm)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )

        chunks = text_splitter.split_documents([doc])

        for chunk in chunks:
            chunk.page_content = f"Résumé: {summary}\n\n{chunk.page_content}"
            summarized_chunks.append(chunk)

    print(f"Number of text splits generated: {len(summarized_chunks)}")

    embedding_db = Chroma.from_documents(
        summarized_chunks, embedding_model, persist_directory=embedding_directory
    )

    print("Embeddings completed")
