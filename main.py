import streamlit as st
import pandas as pd
import numpy as np
from streamlit_chat import message
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import json
import os
import textwrap

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

PINECONE_ENV = 'us-west1-gcp-free'
PINECONE_INDEX_NAME = 'bv-index-1536'

st.set_page_config(page_title="BLZ Search Demo", page_icon=":robot:")
st.title('BLZ Search Demo')

question_container = st.container()
details_container  = st.container()
output_container   = st.container()

details_container.markdown(f'a={len(OPENAI_API_KEY)}  p={len(PINECONE_API_KEY)}', unsafe_allow_html=True)

@st.cache_resource
def create_embeddings():
    embeddings = OpenAIEmbeddings()
    return embeddings

@st.cache_resource
def pinecone_init():
    pinecone.init(
        api_key    = PINECONE_API_KEY,  # find at app.pinecone.io
        environment= PINECONE_ENV       # next to api key in console
    )

@st.cache_resource 
def load_chain(_embeddings):
    docsearch = Pinecone.from_existing_index(PINECONE_INDEX_NAME, _embeddings)
    return docsearch

def get_question():
    return question_container.text_input("You: ", "", key="input")
    
embeddings = create_embeddings()    
chain = load_chain(embeddings)
pinecone_init()


user_input = get_question()

if user_input:
    docs = chain.similarity_search_with_score(user_input, k = 3)

    for i in range(len(docs)):
        d = docs[i]
        content = textwrap.fill(d[0].page_content, 100)
        doc_ref = os.path.basename(d[0].metadata['source'])
        score   = d[1]
        
        details_container.markdown(f'<b>{doc_ref}</b> [score={score}]:', unsafe_allow_html=True)
        details_container.markdown(content, unsafe_allow_html=True)
