from scraper import scrape, extract_txt_from_dir
from embedder import embed
from embedder_raptor import embed_raptor
from embedder_summary import embed_summary
from chat import startChat
import os
import warnings


os.environ["TESSDATA_PREFIX"] = r"C:\\Program Files\\Tesseract-OCR\\tessdata"

# todo: read the conf from a json file
config = {
    "llm_ollama": "Mistral-NeMo-for-RAG:latest",
    "documents_path": r"C:\Users\bertr\PycharmProjects\TestRAG\data\wetransfer_pieces-we-transfert_2023-10-27_1533",
    "content_path": r"C:\Users\bertr\PycharmProjects\TestRAG\simple-rag-lmstudio-main\content",
    # "content_path": r"C:\Users\bertr\PycharmProjects\TestRAG\simple-rag-lmstudio-main\content_test",
    "pytesseract_path": r"C:/Program Files/Tesseract-OCR/tesseract.exe",
    "embedding_model": "Lajavaness/bilingual-embedding-large-8k",
    "FORCE_SCRAPER": False,
    "FORCE_EMBED": True,
    "EMBED_RAPTOR": True,
    "EMBED_SUMMARY": False,
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "n_levels_raptor": 3,
    "ui": "gradio",
}


documents_path = config["documents_path"]
content_path = config["content_path"]
pytesseract_path = config["pytesseract_path"]
embedding_model = config["embedding_model"]
FORCE_SCRAPER = config["FORCE_SCRAPER"]
FORCE_EMBED = config["FORCE_EMBED"]
EMBED_SUMMARY = config["EMBED_SUMMARY"]
EMBED_RAPTOR = config["EMBED_RAPTOR"]
llm_ollama = config["llm_ollama"]

if FORCE_SCRAPER:
    extract_txt_from_dir(documents_path, content_path, pytesseract_path)

if FORCE_EMBED:
    if EMBED_RAPTOR:
        embed_raptor(config)
        if EMBED_SUMMARY:
            warnings.warn("EMBED_SUMMARY not compatible yet with RAPTOR!!")
    else:
        if EMBED_SUMMARY:
            embed_summary(config)
        else:
            embed(config)

startChat(config)
