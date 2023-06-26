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
from redlines import Redlines  

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 

how_it_work = """\
Put your text here to check your German.
Press Ctrl+Enter and wait for Gpt advice. 
"""

check_prompt_template = """/
Hi, I want to check my German. 
I will write one sentence - please correct me if I'm wrong.
If there are errors please explain me step by step why and what is wrong.
Provide answer in JSON format with fields:
- correct - correct sentence in German
- explaination - list of explaination
###
Text:
{text}
"""

rus_prompt_template = """/
You are the best translator from English to Russuan.
Please translate provided text into Russuan.
###
Text:
{text}
"""


st.set_page_config(page_title="Gpt German Gramma Checker", page_icon=":robot:")
st.title('Gpt German Gramma Checker')

header_container   = st.container()
text_container     = st.container()
correct_container  = st.container()
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

def create_check_chain(_llm):
    prompt = PromptTemplate.from_template(check_prompt_template)
    chain  = LLMChain(llm=_llm, prompt = prompt)
    return chain

def create_rus_chain(_llm):
    prompt = PromptTemplate.from_template(rus_prompt_template)
    chain  = LLMChain(llm=_llm, prompt = prompt)
    return chain
    
llm = load_llm()
check_chain = create_check_chain(llm)
rus_chain   = create_rus_chain(llm)

user_input = get_text()

if user_input:
    
    result = check_chain.run(text=user_input)
    result_json = json.loads(result)
    
    correct = result_json["correct"]
    diff = Redlines(user_input, correct) 
    correct_container.markdown(diff.output_markdown, unsafe_allow_html=True)
    
    e_rus = []
    for e in result_json["explanation"]:
        result_rus = rus_chain.run(text=e)
        e_rus.append(f'<li>{result_rus}</li>')
    full_explain_rus = "\n".join(e_rus)
    explain_container.markdown(full_explain_rus, unsafe_allow_html=True)
