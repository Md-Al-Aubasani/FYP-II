from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage




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

api_key = "YOUR_API_KEY"

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

system_prompt = """
You are a helpful, respectful and honest medical bot. Always answer as
helpfully as possible, while being safe.

If a question does not make any sense, or is not factually coherent, explain
why instead of answering something not correct. If you don't know the answer
to a question, please don't share false information.

Answer the following question based only on the user info:

<user_info>
{user_info}
</user_info>

You can look at the medical context to see if it is relevant and see if it can help the medical suggestion you're making.
<context>
{context}
</context>
"""


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! upload your patient text files and ask me anything about them!"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey, there!"]

def conversation_chat(query, chain, history, user_vector_store):
    # retrieving releavant user context
    docs_and_scores = user_vector_store.similarity_search_with_score(query)
    user_info = [doc.page_content for doc, score in docs_and_scores]
    result = chain.invoke({"input": query, "chat_history": history, "user_info": user_info})
    history.extend(
      [
        HumanMessage(content=query),
        AIMessage(content=result["answer"]),
      ]
    )
    return result["answer"]

def display_chat_history(chain, user_vector_store):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your patient text file", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'], user_vector_store)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="pixel-art")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

def create_conversational_chain(vector_db):
    # Create llm
    model = ChatMistralAI(mistral_api_key=api_key)

    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        model, vector_db.as_retriever(search_kwargs={"k": 2}), contextualize_q_prompt
    )
    
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    
    
    
    # Create a retrieval chain to answer questions
    document_chain = create_stuff_documents_chain(model, qa_prompt)
    
    
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
    #                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    #                                              memory=memory)
    return retrieval_chain

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
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


        # Create vector store
        user_vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Medical vector store
        vector_db = FAISS.load_local('/content/drive/MyDrive/faiss_index_full', embeddings, allow_dangerous_deserialization=True)

        # Create the chain object
        chain = create_conversational_chain(vector_db)

        
        display_chat_history(chain, user_vector_store)

if __name__ == "__main__":
    main()
