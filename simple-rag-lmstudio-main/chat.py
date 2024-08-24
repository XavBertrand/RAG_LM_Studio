from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import prompt
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import textwrap
import gradio as gr
import streamlit as st
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
)
from langchain_community.vectorstores import Chroma
import os
import tkinter as tk
from tkinter import filedialog
import zipfile

import asyncio
from langchain_community.callbacks import get_openai_callback

from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate


os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OPENAI_API_KEY"] = "dummy-key"


def get_prompt_EN():
    return """Use the following Context information to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
### Instruction:
Context: {context}
User Question: {question}
###
Response:
"""


def get_prompt_FR():
    return """Tu es un avocat qui cherche à montrer la vérité. Tu réponds systématiquement en français. Utilise les informations de Contexte suivantes pour répondre à la question de l’utilisateur. Si tu ne connais pas la réponse, dis simplement que tu ne sais pas, n’essaie pas d’inventer une réponse.
### Instruction :
Contexte : {context}
Question de l’utilisateur : {question}
###
Réponse :
"""


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split("\n")

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = "\n".join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    if not llm_response:
        return "Please enter a question"
    print(wrap_text_preserve_newlines(llm_response["result"]))
    # print("\n\nSources:")
    # for source in llm_response["source_documents"]:
    #     print(source.metadata["source"])
    response = llm_response["result"]
    response = response.split("### Response")[0]
    return response


def chatbot_response(user_input):
    # Votre logique de traitement du texte ici
    return "Réponse du chatbot"


# Fonction pour traiter le fichier zip
def process_zip_file(uploaded_zip):
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall("extracted_files")
    st.success("Fichier zip traité avec succès!")


# Fonction pour lancer le programme de traitement
def run_processing_program():
    st.info("Lancement du programme de traitement...")
    # Ajouter ici le code de traitement spécifique
    st.success("Programme de traitement terminé!")


def gradio_ui(runChain):
    chatbot_interface = gr.ChatInterface(
        runChain,
        title="ActionAvocatsGPT",
    )
    chatbot_interface.queue()
    chatbot_interface.launch(share=False, debug=True)


def streamlit_ui(runChain):
    # Configuration de l'interface Streamlit
    st.set_page_config(layout="wide")

    # Bandeau de gauche
    st.sidebar.header("Options")
    menu_options = ["Option 1", "Option 2", "Option 3"]
    selected_option = st.sidebar.selectbox("Sélectionner un élément:", menu_options)

    uploaded_file = st.sidebar.file_uploader("Charger un fichier zip", type="zip")

    if st.sidebar.button("Charger le fichier"):
        if uploaded_file is not None:
            process_zip_file(uploaded_file)
        else:
            st.sidebar.error("Veuillez charger un fichier zip.")

    if st.sidebar.button("Lancer le traitement"):
        run_processing_program()

    # Fenêtre principale pour le chatbot
    st.title("ActionAvocatsGPT")

    # Initialisation de l'état des messages du chatbot
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Initialisation de l'état de l'entrée utilisateur
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""

    # Entrée de texte pour l'utilisateur
    user_input = st.text_input("Vous:", key="user_input")

    if st.button("Envoyer"):
        if user_input:
            # Récupérer l'historique des messages
            history = [
                (msg["role"], msg["content"]) for msg in st.session_state["messages"]
            ]

            # Obtenir la réponse du chatbot
            response = runChain(user_input, history)

            # Ajouter la requête de l'utilisateur et la réponse du chatbot aux messages
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "bot", "content": response})

            # Effacer l'entrée utilisateur sans affecter la clé initiale
            st.session_state["user_input"] = ""

    # Affichage des messages du chatbot
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            st.markdown(f"**Vous:** {message['content']}")
        else:
            st.markdown(f"**Chatbot:** {message['content']}")


def startChat(embedding_model="dangvantuan/sentence-camembert-base", config=None):

    llm = ChatOllama(model=config["llm_ollama"])

    if config["RAPTOR"]:
        embedding_directory = r"C:\Users\bertr\PycharmProjects\TestRAG\simple-rag-lmstudio-main\content\chroma_db_raptor"
    elif config["EMBED_SUMMARY"]:
        embedding_directory = r"C:\Users\bertr\PycharmProjects\TestRAG\simple-rag-lmstudio-main\content\chroma_db_summary"
    else:
        embedding_directory = r"C:\Users\bertr\PycharmProjects\TestRAG\simple-rag-lmstudio-main\content\chroma_db"

    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda", "trust_remote_code": True},
    )
    embedding_db = Chroma(
        persist_directory=embedding_directory, embedding_function=embedding_model
    )

    prompt_template = get_prompt_FR()

    llama_prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": llama_prompt}

    # "similarity" or "mmr"
    # retriever = embedding_db.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    retriever = MultiQueryRetriever.from_llm(
        retriever=embedding_db.as_retriever(search_type="mmr", search_kwargs={"k": 10}),
        llm=llm,
    )

    # create the chain to answer questions
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True,
    )

    def run_chain(query, history):
        return process_llm_response(qa_chain(query))

    ui = config["ui"]  # gradio or streamlit
    if ui == "gradio":
        gradio_ui(run_chain)
    elif ui == "streamlit":
        streamlit_ui(run_chain)
