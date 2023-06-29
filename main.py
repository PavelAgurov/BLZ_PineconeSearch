import streamlit as st
import pandas as pd
import numpy as np
from streamlit_chat import message
import pinecone
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
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

PINECONE_ENV = 'us-west1-gcp-free'
PINECONE_INDEX_NAME = 'bv-index-1536'

how_it_work = """\
Ask your question in your language. 
Gpt will extract your question in German, search in BLZ index and return result in your original language as Gpt-summary of articles.
Please note - not all articles can be relevant, it's only similarity search (first score). Second score is defined by Gpt.
"""

question_prompt_template = """/
You are the best German translator.
You should translate remove all not significant words from provided question and translate it to German.
You should provide result in JSON format with fields:
- lang (language of original question)
- question (extracted and translated question)
Be sure that result is real JSON.
Question: {question}
"""

relevant_prompt_template = """/
You are the best linguist who can compare texts.
You should understand if provided texts are relevant to the provided question.
Relevance score is a number from 0 till 1. 0 means "not relevant", 1 means "relevant".
###
Provide result as array in JSON format with fields:
- text_name - name of text
- explanation - explanation in English why text is relevant to the question
- score - score how text is relevant to the question
###
Question: {question}
{texts}
"""

summary_prompt_template = """/
You are the best German linguist.
You must create a concise and clear answer to the provided question as summary from the provided texts.
You must ONLY use the provided texts. You must NOT use previous knowledge.
Give only a direct answer to the question, do not add irrelevant details.
Question: {question}
{texts}
"""

translation_prompt_template = """/
You are the best German linguist.
You should translate provided text from German to {lang} language.
{text}
"""

st.set_page_config(page_title="BLZ Search Demo", page_icon=":robot:")
st.title('BLZ Search Demo')

header_container   = st.container()
question_container = st.container()
details_container  = st.container()
output_container   = st.container()

header_container.markdown(how_it_work,
 unsafe_allow_html=True)

@st.cache_resource
def create_embeddings():
    embeddings = OpenAIEmbeddings()
    return embeddings

def pinecone_init():
    pinecone.init(
        api_key    = PINECONE_API_KEY,  # find at app.pinecone.io
        environment= PINECONE_ENV       # next to api key in console
    )

@st.cache_resource 
def load_chain(_embeddings):
    docsearch = Pinecone.from_existing_index(PINECONE_INDEX_NAME, _embeddings)
    return docsearch

@st.cache_resource
def load_llm():
    llm = ChatOpenAI(
        openai_api_key= OPENAI_API_KEY,
        model_name  = "gpt-3.5-turbo", 
        temperature = 0, 
        max_tokens  = 500
    )
    return llm

def get_question():
    return question_container.text_input("Your question: ", "", key="input")

def get_counter():
    return question_container.number_input("Count of examples: ", min_value=1, max_value=6, value=3, key="example_counter")
    
def create_question_chain(_llm):
    prompt = PromptTemplate.from_template(question_prompt_template)
    chain  = LLMChain(llm=_llm, prompt = prompt)
    return chain

def create_relevant_chain(_llm):
    prompt = PromptTemplate.from_template(relevant_prompt_template)
    chain  = LLMChain(llm=_llm, prompt= prompt)
    return chain

def create_summary_chain(_llm):
    prompt = PromptTemplate.from_template(summary_prompt_template)
    chain = LLMChain(llm=_llm, prompt= prompt)
    return chain

def create_translation_chain(_llm):
    prompt = PromptTemplate.from_template(translation_prompt_template)
    chain = LLMChain(llm=_llm, prompt= prompt)
    return chain
    
def extract_question(question_chain, txt):
    return question_chain.run(question=txt)
    
pinecone_init()
embeddings = create_embeddings()    
chain = load_chain(embeddings)

llm = load_llm()
question_chain    = create_question_chain(llm)
relevant_chain    = create_relevant_chain(llm)
summary_chain     = create_summary_chain(llm)
translation_chain = create_translation_chain(llm)

user_input     = get_question()
count_examples = get_counter()

if user_input:
    
    extracted_question = extract_question(question_chain, user_input)

    error_question = False   
    try:
        extracted_question_json = json.loads(extracted_question)
    except Exception as error:
        error_question = True
        output_container.markdown(f'Looks like it\'s not a question.\nError JSON: {extracted_question}.\nError: {error}\n{traceback.format_exc()}', unsafe_allow_html=True)
        
    if not error_question:
        try:
            original_lang = extracted_question_json['lang']
            de_question   = extracted_question_json['question']
            details_container.markdown(f'[Was in {original_lang}] {de_question}', unsafe_allow_html=True)

            docs = chain.similarity_search_with_score(de_question, k = count_examples)

            content_list = []
            text_index = 1
            for i in range(len(docs)):
                d = docs[i]
                content = d[0].page_content
                content_list.append(f'Text-{text_index}:\n{content}')
                text_index = text_index+1

            texts = '\n'.join(content_list)

            result_relevant = relevant_chain.run(
              question = de_question,
              texts = texts
            )

            try:
                result_relevant_json = json.loads(result_relevant)
                
                content_list = []
                text_index = 1
                for i in range(len(docs)):
                    d = docs[i]
                    content   = textwrap.fill(d[0].page_content, 100)
                    doc_ref   = os.path.basename(d[0].metadata['source'])
                    score     = result_relevant_json[i]["score"]
                    sim_score = d[1]
                    explanation = result_relevant_json[i]["explanation"].replace(f'Text-{text_index}', 'This text')
                    
                    details_container.markdown(f'<b>{doc_ref}</b> [score={sim_score:1.2f}/{score:1.2f}]:', unsafe_allow_html=True)
                    details_container.markdown(content, unsafe_allow_html=False)
                    details_container.markdown(f'<i>{explanation}</i>', unsafe_allow_html=True)
                    
                    if score >= 0.5:
                        content_list.append(f'Text-{text_index}:\n{content}')
                        text_index = text_index + 1

                if len(content_list) == 0:
                    output_container.markdown('>There is no answer', unsafe_allow_html=True)
                else:
                    texts = '\n'.join(content_list)
                    
                    result_summary = summary_chain.run(
                      question = de_question,
                      texts = texts
                    )
                    
                    if original_lang != 'German':
                        result_summary = translation_chain.run(
                          lang = original_lang,
                          text = result_summary
                        )
                
                    output_container.markdown(f'>{result_summary}', unsafe_allow_html=True)
                    
            except Exception as error:
                output_container.markdown(f'>[ERROR]\n{result_relevant}. \nError: {error}\n{traceback.format_exc()}', unsafe_allow_html=True)
                    
        except Exception as error:
            output_container.markdown(f'[ERROR] {error}\n{traceback.format_exc()}', unsafe_allow_html=True)
        

        
    

