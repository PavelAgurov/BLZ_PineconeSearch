import streamlit as st
import pandas as pd
import langchain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser
from langchain.cache import SQLiteCache
import json
import os
import re
from redlines import Redlines  
import datetime
from utils import streamlit_hack_disable_textarea_submit, streamlit_hack_remove_top_space

SENTENCE_SUFFIX_LIST = ['.', '!', '?']

how_it_work = """\
Enter your translation of proposed text, click Check and wait for Gpt validation and advice.
Plese note - it's still POC.
"""

generation_template = """/
Hello! I learn {lang_learn}.
Make me a random sentence in {lang_my} for translation into {lang_learn} and translate all the words in it, 
but not the sentence itself. All nouns must have an article (for example "der Ort", "das Essen").
Use this {random} value for seed randomization and generate different sentences.
{level_and_type}
Provide answer in JSON format:
{{
    "proposed_sentence" : "proposed sentence in {lang_my}",
    "words":{{
        "nouns":[
            {{
                "word": "noun from {lang_my} sentence",
                "infinitive": "noun from {lang_my} sentence in infinitive and singular form", 
                "translation" : "translation into {lang_learn} of infinitive and singular form with article if article is needed"
            }}
        ],
        "adjectives":[
            {{
                "word": "adjective from {lang_my} sentence",
                "infinitive": "adjective from {lang_my} sentence in infinitive and singular form", 
                "translation" : "translation into {lang_learn} of infinitive and singular form"
            }}
        ],
        "verbs":[
            {{
                "word": "verb from {lang_my} sentence",
                "infinitive": "verb from {lang_my} sentence in infinitive and singular form", 
                "translation" : "translation into {lang_learn} of infinitive and singular form"
            }}
        ],
        "prepositions":[
            {{
                "word": "preposition from {lang_my} sentence",
                "translation" : "translation into {lang_learn} of infinitive and singular form"
            }}
        ],
        "other":[
            {{
                "word": "word from {lang_my} sentence",
                "infinitive": "word from {lang_my} sentence in infinitive and singular form", 
                "translation" : "translation into {lang_learn} of infinitive and singular form"
            }}
        ]
    }}

}}
"""
# фильм	the films	существительное - разделить на слова, перевести в инфинитив и переводить отдельно каждое
# В магазине было много красивых цветов.
# цвет	the flowers	существительное

check_prompt_template = """/
Hi, I want to check my {lang_learn}.
I have translation sentence (separated by XML tags) from {lang_my} into {lang_learn} (separated by XML tags).
Please correct me if my translation is wrong and if there are errors please explain me step by step all my mistakes.
Do not check original sentence, only check my translation.
Don't try to make up an errors, only provide me information about my mistakes in this sentence.
All explanations should be provided in {lang_my}.

Provide answer in JSON format:
{{
    "correct" : "correct sentence in {lang_learn}",
    "errors_explanations":[
        "my mistakes explanation",
        "the same explanation translated into {lang_my}"
    ]
}}
Be sure that result is valid JSON.

<input_sentence>{input_sentence}</input_sentence>
<translation>{translation}</translation>
"""

st.set_page_config(page_title="Gpt Language Trainer", layout="wide")
st.title("Gpt Language Trainer")

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

@st.cache_data
def get_default_gpt_key():
    if "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"]
    else:
        return ""

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

query_params = st.experimental_get_query_params()
from_lang_default_value = "Russian"
if "from" in query_params:
    from_lang_default_value = query_params["from"][0].strip("\"")
to_lang_default_value = "German"
if "to" in query_params:
    to_lang_default_value = query_params["to"][0].strip("\"")

header_container = st.container()
tab_main, tb_settings, tab_debug = st.tabs(["Main", "Settings", "Debug"])

with tab_main:
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

with tb_settings:
    gpt_key_input    = st.text_input("Your Gpt key: ",  value = get_default_gpt_key(), type='password')
    gpt_key_info     = st.info("Your key is only used in your browser. If you refresh the page - enter the key again.")
    lang_my_input    = st.text_input("I speak: ",  value = from_lang_default_value)
    lang_learn_input = st.text_input("I learn: ", value = to_lang_default_value)
    level_input      = st.selectbox("Level:", key="slevel", options=["Simple", "Medium", "Advanced"], index=1)

with tab_debug:
    init_json_container     = st.expander(label="JSON init")
    validate_json_container = st.expander(label="JSON validation")

with st.sidebar:
    info_container  = st.container()
    type_input      = st.selectbox("Sentence type:", key="stype", options=["Statement", "Questions"], index=1)
    words_container = st.expander(label="Help me with words")
    error_container = st.empty()

header_container.markdown(how_it_work, unsafe_allow_html=True)
streamlit_hack_remove_top_space()
streamlit_hack_disable_textarea_submit()

if not gpt_key_input:
    info_message = '<p style="color:white; background-color:red">Enter your Gpt key on Settings tab</p>'
    info_container.markdown(info_message, unsafe_allow_html=True)

def get_level_and_type(level, atype):
    if level == "Simple":
        return f"Sentence should be very simple - noun, verb and adjective and be {atype}."
    elif level == "Medium":
        return f"Sentence should have medium complexity and have maximum 10 words and be {atype}."
    elif level == "Advanced":
        return f"Sentence should be complex and have minumum 20 words and subordinate clause and be {atype}."

def get_fixed_json(text : str) -> str:
    text = text.replace(", ]", "]").replace(",]", "]").replace(",\n]", "]")
    open_bracket = min(text.find('['), text.find('{'))
    if open_bracket == -1:
        return text
            
    close_bracket = max(text.rfind(']'), text.rfind('}'))
    if close_bracket == -1:
        return text
    return text[open_bracket:close_bracket+1]

def remove_double_spaces(input):
    return re.sub(' +', ' ', input)

if not gpt_key_input:
    st.stop()

langchain.llm_cache = SQLiteCache()
llm_random = ChatOpenAI(
        openai_api_key= gpt_key_input,
        model_name  = "gpt-3.5-turbo", 
        temperature = 0.9, 
        max_tokens  = 1000
)
llm_fixed = ChatOpenAI(
        openai_api_key= gpt_key_input,
        model_name  = "gpt-3.5-turbo", 
        temperature = 0, 
        max_tokens  = 1000
)
generation_prompt  = PromptTemplate.from_template(generation_template)
generation_chain  = generation_prompt | llm_random | StrOutputParser()
validation_prompt = PromptTemplate.from_template(check_prompt_template)
validation_chain  = LLMChain(llm= llm_fixed, prompt = validation_prompt)

now_str = datetime.datetime.utcnow().strftime('%F %T.%f')[:-3]

run_check = st.session_state[SESSION_RUN_CHECK]
saved_user_input = st.session_state[SESSION_SAVED_USER_INPUT]
generated_sentence = st.session_state[SESSION_SAVED_SENTENCE]

# not generated yet and it's not checking
if not generated_sentence and not run_check:
    status_container.markdown('Generate sentence...')
    generated_sentence_result = generation_chain.invoke({
            "level_and_type" : get_level_and_type(level_input, type_input), 
            "lang_learn" : lang_learn_input,
            "lang_my": lang_my_input,
            "random" : now_str
        })
    status_container.markdown(' ')
    init_json_container.markdown(generated_sentence_result)
    try:
        proposed_sentence_json = json.loads(get_fixed_json(generated_sentence_result))
        proposed_sentence = proposed_sentence_json['proposed_sentence']
        proposed_words    = proposed_sentence_json['words']
        proposed_words_list = []
        for w in proposed_words['nouns']:
            proposed_words_list.append([ w['infinitive'], w["translation"]])
        for w in proposed_words['adjectives']:
            proposed_words_list.append([ w['infinitive'], w["translation"]])
        for w in proposed_words['verbs']:
            proposed_words_list.append([ w['infinitive'], w["translation"]])
        for w in proposed_words['other']:
            proposed_words_list.append([ w['infinitive'], w["translation"]])
        df_words = pd.DataFrame(proposed_words_list, columns= ['Word', 'Translation'])
        st.session_state[SESSION_SAVED_WORDS_DF] = df_words
        st.session_state[SESSION_SAVED_SENTENCE] = proposed_sentence
    except Exception as error:
        error_container.markdown(f'{generated_sentence_result}\n{error}')

words_container.dataframe(st.session_state[SESSION_SAVED_WORDS_DF], use_container_width=True, hide_index=True)
original_container.markdown(st.session_state[SESSION_SAVED_SENTENCE])

# if we have user input and it's checking - run validation
if saved_user_input and run_check:
    status_container.markdown('Validation...')
    validation_result = validation_chain.run(
            input_sentence = st.session_state[SESSION_SAVED_SENTENCE],  
            translation    = saved_user_input, 
            lang_learn     = lang_learn_input,
            lang_my        = lang_my_input
    )
    status_container.markdown(' ')
    validate_json_container.markdown(validation_result)

    try:
        result_json = json.loads(get_fixed_json(validation_result))
        
        correct : str = remove_double_spaces(result_json["correct"]).strip()
        explanation = result_json["errors_explanations"]

        st.session_state[SESSION_CORRECT_SENTENCE] = correct
        st.session_state[SESSION_EXPLANATION] = explanation

    except Exception as error:
        explain_container.markdown(f'Error: [{validation_result}]\n{error}', unsafe_allow_html=True)

    st.session_state[SESSION_RUN_CHECK] = False

correct = st.session_state[SESSION_CORRECT_SENTENCE]
explanation = st.session_state[SESSION_EXPLANATION]

if correct and saved_user_input:
    correct = remove_double_spaces(correct).strip()
    saved_user_input = remove_double_spaces(saved_user_input).strip()

    correct_suffix = correct[-1]
    user_suffix    = saved_user_input[-1]
    if correct_suffix in SENTENCE_SUFFIX_LIST:
        if correct_suffix != user_suffix:
            if user_suffix in SENTENCE_SUFFIX_LIST:
                saved_user_input = saved_user_input[:-1]
            saved_user_input = saved_user_input + correct_suffix

    diff = Redlines(saved_user_input, correct)
    correct_container.markdown(diff.output_markdown, unsafe_allow_html=True)

    try:
        if len(explanation) > 0:
            if lang_my_input.lower() != "english":
                #explanation_translated = '\n'.join([f'<li>{e[2]}</li>' for e in explanation])
                explanation_translated = explanation[1]
                explain_container.markdown(explanation_translated, unsafe_allow_html=True)
            else:
                #explanation_english = '\n'.join([f'<li>{e[1]}</li>' for e in explanation])
                explanation_english = explanation[0]
                explain_container.markdown(explanation_english, unsafe_allow_html=True)
        else:
            correct_message = '<p style="color:green;">Correct!</p>'
            explain_container.markdown(correct_message, unsafe_allow_html=True)
    except Exception as error:
        explain_container.markdown(f'Error: [{explanation}]\n{error}', unsafe_allow_html=True)

else:
    correct_container.markdown('', unsafe_allow_html=True)
    explain_container.markdown('', unsafe_allow_html=True)

