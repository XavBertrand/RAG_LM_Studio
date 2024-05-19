from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOllama, ChatOpenAI
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

# TODO: Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

# temperature = 0.1  # Use a value between 0 and 2. Lower = factual, higher = creative
# n_gpu_layers = 43  # Change this value based on your model and your GPU VRAM pool.
# n_batch = (
#     1024  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
# )

# Make sure the model path is correct for your system!
llm = ChatOpenAI(openai_api_base="http://localhost:1234/v1", openai_api_key="dummy-key")
# llm = ChatOllama(model="mistral-7b-instruct-v0.2.Q4_K_M:latest")

# n_gpu_layers = (
#     -1
# )  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
# n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
# n_ctx = 2048
#
#
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#
# # Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     # model_path=r"C:\Users\bertr\LLM_GGUF\models\Huggingface\Mixtral-8x7B\mixtral-8x7b-v0.1.Q4_K_M.gguf",
#     model_path=r"C:\Users\bertr\LLM_GGUF\models\Huggingface\Mistral-7B\mistral-7b-instruct-v0.2.Q4_K_M.gguf",
#     n_gpu_layers=n_gpu_layers,
#     n_batch=n_batch,
#     n_ctx=n_ctx,
#     callback_manager=callback_manager,
#     f16_kv=True,
#     verbose=True,  # Verbose is required to pass to the callback manager
# )


## Follow the default prompt style from the OpenOrca-Platypus2 huggingface model card.


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


def select_folder():
    root = tk.Tk()
    root.withdraw()  # Masquer la fenêtre principale
    folder_path = (
        filedialog.askdirectory()
    )  # Ouvrir la boîte de dialogue de sélection de dossier
    if folder_path:
        print("Dossier sélectionné :", folder_path)
        # Exécutez votre script "embedding.py" avec le chemin du dossier sélectionné ici


def startChat(embedding_model="dangvantuan/sentence-camembert-base", raptor=False):

    if raptor:
        embedding_directory = r"C:\Users\bertr\PycharmProjects\TestRAG\simple-rag-lmstudio-main\content\chroma_db_raptor"
    else:
        embedding_directory = r"C:\Users\bertr\PycharmProjects\TestRAG\simple-rag-lmstudio-main\content\chroma_db"

    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda"},
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
        retriever=embedding_db.as_retriever(search_type="mmr", search_kwargs={"k": 5}),
        llm=llm,
    )

    # create the chain to answer questions
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        # return_source_documents=True,
    )

    def runChain(query, history):
        return process_llm_response(qa_chain(query))

    chatbot_interface = gr.ChatInterface(runChain)
    chatbot_interface.queue()
    chatbot_interface.launch(share=False, debug=True)