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
Put your text here and generate the best title for SEO.
Press Ctrl+Enter and wait for Gpt advice. 
"""

seo_prompt_template = """/
You are SEO manager. 
Please generate the best title for provided article for publishing in {channel} and 
explain step by step why it's the best for concrete this channel from SEO point of view.
Provide result in JSON format with fields: title and list of explanations. 
Title and explanations should be in German.
Before answer check that you follow recommendation that you did.
Be sure that result is valid JSON.
###
Text:
{text}
"""

#Explanations should be in English.


st.set_page_config(page_title="BLZ SEO Demo", page_icon=":robot:")
st.title('BLZ SEO Demo')

header_container   = st.container()
text_container     = st.container()
title_container    = st.container()
explain_container  = st.container()

header_container.markdown(how_it_work, unsafe_allow_html=True)

@st.cache_resource
def load_llm():
    llm = ChatOpenAI(
        openai_api_key= OPENAI_API_KEY,
        model_name  = "gpt-3.5-turbo", 
        temperature = 0.5, 
        max_tokens  = 500
    )
    return llm

def create_seo_chain(_llm):
    prompt = PromptTemplate.from_template(seo_prompt_template)
    chain  = LLMChain(llm=_llm, prompt = prompt)
    return chain
    
llm = load_llm()
seo_chain = create_seo_chain(llm)

# ---------- UI
user_input = text_container.text_area("Article text: ", key="input")
channel    = text_container.selectbox('Channel:', ('Web', 'Facebook', 'Twitter'))

# ---------- Processing
if user_input:
    
    result = seo_chain.run(text= user_input, channel = channel)

    try:
        result_json = json.loads(result)
        
        title_container.markdown(f"Title:<br/><b>{result_json['title']}</b>", unsafe_allow_html=True)
            
        list_li = []
        for e in result_json['explanations']:
            list_li.append(f'<li>{e}')
        explain_result = '\n'.join(list_li)
        explain_container.markdown(explain_result, unsafe_allow_html=True)
    
    except Exception as error:
        explain_container.markdown(f'Error JSON: [{result}]', unsafe_allow_html=True)

        

