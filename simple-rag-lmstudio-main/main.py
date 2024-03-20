from scraper import scrape, extract_txt_from_dir
from embedder import embed
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

# embedding_model = "dangvantuan/sentence-camembert-base"
embedding_model = "dangvantuan/sentence-camembert-large"

FORCE = True
if FORCE:
    # scrape(start_url, start_depth)
    extract_txt_from_dir(documents_path, content_path, pytesseract_path)
    embed(embedding_model)

startChat(embedding_model)
