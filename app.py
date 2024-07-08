from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain





import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.llms import LlamaCpp
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile


api_key = 'API_KEY'


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! upload your patient text files and ask me anything about them!"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey, there!"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your patient text file", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="pixel-art")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

def create_conversational_chain(vector_store):
    # Create llm
    llm = ChatMistralAI(mistral_api_key=api_key)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

def main():
    # Initialize session state
    initialize_session_state()
    st.title("Medical Advice ChatBot using Mistral-8x7B-Instruct :male-doctor:")
    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)


    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)


        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Create the chain object
        chain = create_conversational_chain(vector_store)

        
        display_chat_history(chain)

if __name__ == "__main__":
    main()
