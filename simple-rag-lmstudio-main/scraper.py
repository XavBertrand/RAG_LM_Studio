import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import PyPDF2
import fitz
import pytesseract
from PIL import Image
import glob
from docx import Document
from docx2txt import process
import shutil
import tqdm

# Set for storing already visited urls
visited_urls = set()

data_directory = "./content"


def get_page_content(url):
    """
    Returns the content of the webpage at `url`
    """
    response = requests.get(url)
    return response.text


def get_all_links(content, domain):
    """
    Returns all valid links on the page
    """
    soup = BeautifulSoup(content, "html.parser")
    links = soup.find_all("a")
    valid_links = []

    for link in links:
        href = link.get("href")
        if (
            href != None
            and not href.startswith("..")
            and href != "#"
            and not href.startswith("#")
        ):
            if href.startswith("http"):
                if href.startswith(domain):
                    print("Following", href)
                    valid_links.append(href)
            else:

                print("Following", strip_after_last_hash(href))
                valid_links.append(domain + "/" + strip_after_last_hash(href))
    return valid_links


def strip_after_last_hash(url):
    """
    Strips off all characters after the last "#" in `url`,
    if "#" does not have a "/" character before it.
    """
    hash_index = url.rfind("#")
    if hash_index > 0 and url[hash_index - 1] != "/":
        return url[:hash_index]
    else:
        return url


def write_to_file(url, content):
    """
    Write the content to a text file with the name as the URL
    """
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    filename = (
        data_directory
        + "/"
        + url.replace("/", "_").replace(":", "_").replace("?", "_")
        + ".txt"
    )
    with open(filename, "w", encoding="utf-8") as f:
        lines = content.split("\n")
        non_blank_lines = [line for line in lines if line.strip() != ""]
        f.write("\n".join(non_blank_lines))


def scrape(url, depth):
    """
    Scrapes the webpage at `url` up to a certain `depth`
    """
    scheme = urlparse(url).scheme  # Get the scheme
    domain = urlparse(url).netloc  # Get base domain
    path = os.path.dirname(urlparse(url).path)  # Get base path excluding the last part

    print("URL", url)
    if depth == 0 or url in visited_urls:
        return

    visited_urls.add(url)

    print(f"Scraping: {url}")
    content = get_page_content(url)
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text()
    write_to_file(url, text)

    links = get_all_links(content, scheme + "://" + domain + path)

    for link in links:
        scrape(link, depth - 1)


def extract_images_from_docx(docx_path, content_dir):
    # Extraire les images du fichier .docx et les enregistrer

    _, ff = os.path.split(docx_path)
    save_dir = os.path.join(content_dir, ff.replace(" ", "_").replace(".", "_"))

    if os.path.isdir(save_dir):
        save_dir = save_dir + "_"

    os.makedirs(save_dir)

    _ = process(docx_path, save_dir)
    list_img = glob.glob(os.path.join(save_dir, "*"))

    return [os.path.join(save_dir, img) for img in list_img]


def extract_txt_from_dir(dir_path, content_path, pytesseract_path):

    files = glob.glob(os.path.join(dir_path, "**", "*.*"), recursive=True)

    if os.path.isdir(content_path):
        shutil.rmtree(content_path)

    os.makedirs(content_path)

    # Parcourir tous les fichiers du dir_path
    for file in tqdm.tqdm(files):

        loc_path, filename = os.path.split(file)

        if filename.endswith(".pdf"):
            # Ouvrir le fichier PDF
            with open(file, "rb") as fid:
                # Créer un lecteur PDF
                reader = PyPDF2.PdfReader(fid)
                # Extraire le texte de chaque page
                text = ""
                for page in range(len(reader.pages)):
                    text += reader.pages[page].extract_text()

                if text == "":
                    # the pdf content is probably scans
                    doc = fitz.open(file)

                    tmp_dir = os.path.join(content_path, "tmp_dir")
                    if os.path.isdir(tmp_dir):
                        shutil.rmtree(tmp_dir)
                    os.makedirs(tmp_dir)

                    # Parcourir chaque page du PDF
                    for i in range(len(doc)):
                        images = doc.get_page_images(i)
                        for img in images:

                            # Récupérer la liste des images sur la page
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            tmp_file = os.path.join(tmp_dir, f"page_{i}_image_{xref}.jpg")
                            pix.pil_save(tmp_file)
                            image = Image.open(tmp_file)
                            # Appliquer l'OCR à l'image
                            text += pytesseract.image_to_string(image, lang="fra")

                    # Fermer le fichier PDF
                    doc.close()
                    shutil.rmtree(tmp_dir)

                file_txt_name = (
                    filename.replace("/", "_")
                    .replace(":", "_")
                    .replace("?", "_")
                    .replace("\\", "%%")
                    + ".txt"
                )

                with open(
                    os.path.join(content_path, file_txt_name), "w", encoding="utf-8"
                ) as txt_file:
                    txt_file.write(text)

        elif filename.endswith(".jpg") or filename.endswith(".png"):
            # Ouvrir l'image
            image = Image.open(file)
            # Appliquer l'OCR à l'image
            text = pytesseract.image_to_string(image, lang="fra")

            file_txt_name = (
                filename.replace("/", "_")
                .replace(":", "_")
                .replace("?", "_")
                .replace("\\", "%%")
                + ".txt"
            )

            # Enregistrer le texte dans un fichier .txt
            with open(
                os.path.join(content_path, file_txt_name), "w", encoding="utf-8"
            ) as txt_file:
                txt_file.write(text)

        elif filename.endswith(".docx"):
            # Ouvrir le fichier .docx
            doc = Document(file)
            # Extraire le texte de chaque paragraphe
            text = "\n".join([p.text for p in doc.paragraphs])
            # Enregistrer le texte dans un fichier .txt

            file_txt_name = (
                filename.replace("/", "_")
                .replace(":", "_")
                .replace("?", "_")
                .replace("\\", "%%")
                + ".txt"
            )

            with open(
                os.path.join(content_path, file_txt_name), "w", encoding="utf-8"
            ) as txt_file:
                txt_file.write(text)

            # Extraire les images du fichier .docx
            images = extract_images_from_docx(file, content_dir=content_path)

            # Appliquer l'OCR à chaque image
            for i, img_path in enumerate(images):
                image = Image.open(img_path)

                pytesseract.pytesseract.tesseract_cmd = pytesseract_path

                text = pytesseract.image_to_string(image, lang="fra")

                # Enregistrer le texte dans un fichier .txt
                with open(
                    os.path.join(content_path, file_txt_name[:-4] + f"_img{i}.txt"), "w"
                ) as txt_file:
                    txt_file.write(text)
