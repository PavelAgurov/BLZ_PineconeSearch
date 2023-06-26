import streamlit as st
import pandas as pd
import numpy as np
from streamlit_chat import message
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import json
import os
import textwrap
import traceback

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 

how_it_work = """\
Put your text here to check your German.
Press Ctrl+Enter and wait for Gpt advice. 
"""

check_prompt_template = """/
You are German techer. 
I want to check my German. 
I will write sentences - please correct me if I'm wrong and explain why step by step.
Explanation should be in Russian.
###
Text:
{text}
"""


st.set_page_config(page_title="Gpt German Gramma Checker", page_icon=":robot:")
st.title('Gpt German Gramma Checker')

header_container   = st.container()
text_container     = st.container()
explain_container  = st.container()

header_container.markdown(how_it_work, unsafe_allow_html=True)

@st.cache_resource
def load_llm():
    llm = ChatOpenAI(
        openai_api_key= OPENAI_API_KEY,
        model_name  = "gpt-3.5-turbo", 
        temperature = 0, 
        max_tokens  = 500
    )
    return llm

def get_text():
    return text_container.text_area("Text: ", key="input")

def create_chain(_llm):
    prompt = PromptTemplate.from_template(check_prompt_template)
    chain  = LLMChain(llm=_llm, prompt = prompt)
    return chain
    
llm = load_llm()
check_chain = create_chain(llm)

user_input = get_text()

if user_input:
    
    result = check_chain.run(text=user_input)
    explain_container.markdown(result, unsafe_allow_html=True)
