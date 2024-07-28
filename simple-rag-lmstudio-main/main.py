from scraper import scrape, extract_txt_from_dir
from embedder import embed
from embedder_raptor import embed_raptor
from embedder_summary import embed_summary
from chat import startChat
import os

os.environ["TESSDATA_PREFIX"] = r"C:\\Program Files\\Tesseract-OCR\\tessdata"

# Starting URL
# start_url = "https://docs.solace.com/Cloud/Event-Portal/event-portal-lp.htm"
# start_url = "https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html"
# Starting depth
# start_depth = 3

documents_path = r"C:\Users\bertr\PycharmProjects\TestRAG\data\wetransfer_pieces-we-transfert_2023-10-27_1533"
content_path = (
    r"C:\Users\bertr\PycharmProjects\TestRAG\simple-rag-lmstudio-main\content"
)

pytesseract_path = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

embedding_model = "dangvantuan/sentence-camembert-base"
# embedding_model = "dangvantuan/sentence-camembert-large"
# embedding_model = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"  # Large context

FORCE_SCRAPER = False
if FORCE_SCRAPER:
    # scrape(start_url, start_depth)
    extract_txt_from_dir(documents_path, content_path, pytesseract_path)

FORCE_EMBED = False
RAPTOR = False
if FORCE_EMBED:
    if RAPTOR:
        llm_raptor = "Mistral-NeMo-for-RAG:latest"
        embed_raptor(embedding_model, content_path, llm_raptor)
    else:
        # embed(embedding_model, content_path)
        embed_summary(embedding_model, content_path, llm_name="Mistral-NeMo-for-RAG:latest")

startChat(embedding_model, raptor=RAPTOR)
