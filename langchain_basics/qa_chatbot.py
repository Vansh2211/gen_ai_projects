"""
Simple Langchain streamlit app app with groq.
"""


import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

#Page configuration

st.set_page_config(page_title="Simple QA Chatbot with Groq", page_icon=":robot_face:")


with st.sidebar:
    st.header("Settings")

    #apik key
    api_key = st.text_input("GROQ API Key", type="password") 

    model_selection = st.selectbox(
        "Model",
        ["llama-3.1-8b-8192","gemma2-9b-it"],
        index=0
    )

    #clear message
    st.button("Clear Messages")
    st.session_state["messages"] = []
    st.rerun



#Title
st.title("Simple QA Chatbot with Groq")


#Initialize the history
if "messages" not in st.session_state:
    st.session_state["messages"] = []


#Initialize the model

def get_chain(api_key, model_selection):
    if not api_key:
        st.warning("Please enter your GROQ API key in the sidebar.")
        return None
    
    model = ChatGroq(model=model_selection, api_key=api_key, temperature=0.7,streaming=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on the provided context."),
        ("user", "{question}")
    ])

    #create a chain
    chain = prompt | model | StrOutputParser()

    return chain


#get chain
chain = get_chain(api_key, model_selection)


if not chain:
    st.warning("Chain is not initialized. Please check your API key and model selection.")
else:

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    #chat input
    


