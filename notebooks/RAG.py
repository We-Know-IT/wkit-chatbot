from contextlib import asynccontextmanager
import json 
import os
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
import openai
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv(override=True)
api_key = os.environ.get("OPENAI_KEY")

@asynccontextmanager
async def lifespan(app:FastAPI):
    global vectordb, chat_model
    try:
        vectordb = load_vector_db()
        chat_model = load_chatmodel()
        yield
    finally:
        pass

app = FastAPI(lifespan=lifespan)

def load_chatmodel():
    chat_model = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_KEY"),
    model='gpt-3.5-turbo-1106',
    temperature=0.1
    )
    return chat_model
def doc_embedding(embedding_model: str, 
                  model_kwargs: dict={'device':'cpu'}, 
                  encode_kwargs: dict={'normalize_embeddings':True},
                  cache_folder: Optional[str]=None,
                  multi_process: bool=False
                  ) -> HuggingFaceEmbeddings:
    embedder = HuggingFaceEmbeddings(
        model_name = embedding_model,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs,
        cache_folder = cache_folder,
        multi_process = multi_process
    )
    return embedder

def get_API_embedding(text, model):
    embedder = doc_embedding(model)
    embedding = embedder.embed_query(text)
    return embedding

def load_vector_db():
    model = 'mixedbread-ai/mxbai-embed-large-v1'
    embedding_model = doc_embedding(model)
    directory = './e5_ml_db'
    vectordb = Chroma(embedding_function=embedding_model, persist_directory=directory)
    return vectordb


def get_prompt(query: str, vectordb):
    # Retrieve 10 chunks with relevance scores
    context_results = vectordb.similarity_search_with_relevance_scores(query, 10)
    
    # Extract the page_content from each context document
    context = "\n".join([doc.page_content for doc, score in context_results])
    
    # Construct the final prompt
    user_prompt = f"""Svara på följande fråga: {query}.

    Du kan använda följande information för att generera ditt svar:
    {context}"""

    return user_prompt

@app.get("/completion")
def generate_response(query: str):
    global vectordb, chat_model
    system_prompt = """Du är en hjälpsam AI assistent, specialiserad på att svara på frågor om ett IT-konsultbolag som
                heter We Know IT. Du kommer att få frågor samt utvald information, vilken du kan använda
                för att svara på frågan. Svara på svenska."""

    user_prompt = get_prompt(query, vectordb)
    messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_prompt)
    ]

    response = chat_model.invoke(messages)
    return response.content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)