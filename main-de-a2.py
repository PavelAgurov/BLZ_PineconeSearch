import streamlit as st
import pandas as pd
import langchain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import json
import os
from redlines import Redlines  
from langchain.cache import SQLiteCache
import datetime

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 

SENTENCE_SUFFIX_LIST = ['.', '!', '?']

how_it_work = """\
Enter your translation of proposed text, click Check and wait for Gpt validation and advice. 
"""

generation_template = """/
Hello! I am learning {lang}. My level is {level}.
Make me a complicated enough for my level a sentence in Russian for translation into {lang} and translate all the words in it, 
but not the sentence itself.
Use this {random} value for seed randomization and generate different sentences.
Provide answer in JSON format:
{{
    "russian_sentence" : "proposed sentence in Russian",
    "words":[
        ["word": "word in infinitive form", "translation" : "translation into {lang}"}},
        ["word": "word in infinitive form", "translation" : "translation into {lang}"}}
    ]

}}

"""

check_prompt_template = """/
Hi, I want to check my {lang}.
I have translation of Russian sentence (separated by XML tags) into {lang} (separated by XML tags).
Please correct me if my translation is wrong and if there are errors please explain me step by step why and what is wrong.
Please provide as much detail as possible about the mistakes according to the {lang} grammar, but only really relevant to the mistakes made.
Do not check original sentence, only check my translation.
All explanations should be provided in Russian.

Provide answer in JSON format:
{{
    "correct" : "correct sentence in {lang}",
    "errors_explanations":[
        ["wrong word in {lang}", "detailed explanation of this mistake", "the same explanation translated into Russian"],
        ["wrong word in {lang}", "detailed explanation of this mistake", "the same explanation translated into Russian"]
    ]

}}
Be sure that result is valid JSON.

<russian_sentence>{russian_sentence}</russian_sentence>
<translation>{translation}</translation>
"""

SESSION_SAVED_USER_INPUT = 'saved_user_input'
SESSION_RUN_CHECK        = 'run_check'
SESSION_SAVED_SENTENCE   = 'saved_sentence'
SESSION_SAVED_WORDS_DF   = 'saved_words'
SESSION_CORRECT_SENTENCE = 'correct_sentence'
SESSION_EXPLANATION      = 'explanation'

if SESSION_SAVED_USER_INPUT not in st.session_state:
    st.session_state[SESSION_SAVED_USER_INPUT] = ""
if SESSION_RUN_CHECK not in st.session_state:
    st.session_state[SESSION_RUN_CHECK] = False
if SESSION_SAVED_SENTENCE not in st.session_state:
    st.session_state[SESSION_SAVED_SENTENCE] = ""
if SESSION_SAVED_WORDS_DF not in st.session_state:
    st.session_state[SESSION_SAVED_WORDS_DF] = None
if SESSION_CORRECT_SENTENCE not in st.session_state:
    st.session_state[SESSION_CORRECT_SENTENCE] = ""
if SESSION_EXPLANATION not in st.session_state:
    st.session_state[SESSION_EXPLANATION] = ""

def on_check_button_click():
    input_str : str = st.session_state.user_input
    input_str = input_str.strip()
    st.session_state[SESSION_SAVED_USER_INPUT] = input_str
    st.session_state[SESSION_RUN_CHECK] = True

def on_next_button_click():
    st.session_state[SESSION_RUN_CHECK] = False
    st.session_state[SESSION_SAVED_USER_INPUT] = ""
    st.session_state[SESSION_CORRECT_SENTENCE] = ""
    st.session_state[SESSION_EXPLANATION] = ""
    st.session_state[SESSION_SAVED_SENTENCE] = ""
    st.session_state[SESSION_SAVED_WORDS_DF] = None
    st.session_state.user_input = ""

st.set_page_config(page_title="Gpt Language Trainer")
st.title("Gpt Language Trainer")

header_container   = st.container()
status_container   = st.empty()
original_container = st.empty()
input_container    = st.container()
user_input = input_container.text_area("Your translation: ", key="user_input", on_change=on_check_button_click)
correct_container  = st.container().empty()
explain_container  = st.empty()
_  = st.empty().markdown("<hr/>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    check_button = st.button(label= "Check", on_click= on_check_button_click, key="check_button")
with col2:
    next_button  = st.button(label= "Next" , on_click= on_next_button_click , key="next_button" )

with st.sidebar:
    lang_input = st.text_input("Language: ", key="language", value = "German")
    leve_intput = st.selectbox("Level:", key="level", options=["Newcomer", "Novice", "Advanced"], index=2)
    words_container = st.expander(label="Help me with words")
    error_container = st.empty()

header_container.markdown(how_it_work, unsafe_allow_html=True)

def get_fixed_json(text : str) -> str:
    text = text.replace(", ]", "]").replace(",]", "]").replace(",\n]", "]")
    open_bracket = min(text.find('['), text.find('{'))
    if open_bracket == -1:
        return text
            
    close_bracket = max(text.rfind(']'), text.rfind('}'))
    if close_bracket == -1:
        return text
    return text[open_bracket:close_bracket+1]


langchain.llm_cache = SQLiteCache()
llm_random = ChatOpenAI(
        openai_api_key= OPENAI_API_KEY,
        model_name  = "gpt-3.5-turbo", 
        temperature = 0.9, 
        max_tokens  = 1000
)
llm_fixed = ChatOpenAI(
        openai_api_key= OPENAI_API_KEY,
        model_name  = "gpt-3.5-turbo", 
        temperature = 0, 
        max_tokens  = 1000
)
generation_prompt  = PromptTemplate.from_template(generation_template)
generation_chain  = LLMChain(llm= llm_random, prompt= generation_prompt)
validation_prompt = PromptTemplate.from_template(check_prompt_template)
validation_chain  = LLMChain(llm= llm_fixed, prompt = validation_prompt)

now_str = datetime.datetime.utcnow().strftime('%F %T.%f')[:-3]

run_check = st.session_state[SESSION_RUN_CHECK]
saved_user_input = st.session_state[SESSION_SAVED_USER_INPUT]
generated_russian_sentence = st.session_state[SESSION_SAVED_SENTENCE]

# not generated yet and it's not checking
if not generated_russian_sentence and not run_check:
    status_container.markdown('Generate sentence...')
    russian_sentence_result = generation_chain.run(level = leve_intput, lang = lang_input, random = now_str)
    status_container.markdown(' ')
    try:
        russian_sentence_json = json.loads(get_fixed_json(russian_sentence_result))
        russian_sentence = russian_sentence_json['russian_sentence']
        russian_words    = russian_sentence_json['words']
        russian_words_list = []
        for w in russian_words:
            russian_words_list.append([w["word"], w["translation"]])
        df_words = pd.DataFrame(russian_words_list, columns= ['Word', 'Translation'])
        st.session_state[SESSION_SAVED_WORDS_DF] = df_words
        st.session_state[SESSION_SAVED_SENTENCE] = russian_sentence
    except Exception as error:
        error_container.markdown(f'{russian_sentence_result}\n{error}')

words_container.dataframe(st.session_state[SESSION_SAVED_WORDS_DF], use_container_width=True, hide_index=True)
original_container.markdown(st.session_state[SESSION_SAVED_SENTENCE])

# if we have user input and it's checking - run validation
if saved_user_input and run_check:
    status_container.markdown('Validation...')
    validation_result = validation_chain.run(
            russian_sentence= st.session_state[SESSION_SAVED_SENTENCE],  
            translation= saved_user_input, 
            lang = lang_input
    )
    status_container.markdown(' ')

    try:
        result_json = json.loads(get_fixed_json(validation_result))
        
        correct : str = result_json["correct"].strip()
        explanation = result_json["errors_explanations"]

        st.session_state[SESSION_CORRECT_SENTENCE] = correct
        st.session_state[SESSION_EXPLANATION] = explanation

    except Exception as error:
        explain_container.markdown(f'Error: [{validation_result}]\n{error}', unsafe_allow_html=True)

    st.session_state[SESSION_RUN_CHECK] = False

correct = st.session_state[SESSION_CORRECT_SENTENCE]
explanation = st.session_state[SESSION_EXPLANATION]

if correct and saved_user_input:
    correct_suffix = correct[-1]
    user_suffix    = saved_user_input[-1]
    if correct_suffix in SENTENCE_SUFFIX_LIST:
        if correct_suffix != user_suffix:
            if user_suffix in SENTENCE_SUFFIX_LIST:
                saved_user_input = saved_user_input[:-1]
            saved_user_input = saved_user_input + correct_suffix

    diff = Redlines(saved_user_input, correct)
    correct_container.markdown(diff.output_markdown, unsafe_allow_html=True)

    if len(explanation) > 0:
        result_rus = '\n'.join([f'<li>{e[2]}</li>' for e in explanation])
        explain_container.markdown(result_rus, unsafe_allow_html=True)
    else:
        explain_container.markdown("Предложение верное", unsafe_allow_html=True)
else:
    correct_container.markdown('', unsafe_allow_html=True)
    explain_container.markdown('', unsafe_allow_html=True)

