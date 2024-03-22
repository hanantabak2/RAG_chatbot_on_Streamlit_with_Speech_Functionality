# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:45:42 2024

@author: Hanan Tabak
"""

#=================
# Import libraries
#=================
import streamlit as st
import os
import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from io import BytesIO
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI

#------------------------------------------------------------------------------------------------------------------

#=================
# loading PDF, DOCX and TXT files as LangChain Documents
#=================

def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        print(f'Loading {file}')
        data = ''
        pdf_reader = PdfReader(file)
        temp_text = ''
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                temp_text += text
        data += temp_text 
        
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
        data = loader.load()
        
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
        data = loader.load()
        
    else:
        print('Document format is not supported!')
        return None


    return data

#------------------------------------------------------------------------------------------------------------------

#=================
# splitting data in chunks, embedding them, and generate the answer
#=================

# Splitting the data into chunks
def chunk_data(data, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a FAISS vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store


# Design the prompt template and use it to sandwitch the input question, then generate the answer
def ask_and_get_answer(vector_store, q, k=3):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    # Create Prompt
    template = """1- Understand the context of the document first to answer the question at the end. 
    2- If you don't know the answer, just say that you don't know in a professional way.
    3- Don't try to make up an answer.
    4- Respond in the persona of business AI consultant 
    5- Include the page number from which you retreived your content from between parentheses at the tail of the answer.
   
    {context}
    Question: {question}
    Answer: 
    """

    prompt = PromptTemplate(input_variables=["context","question"], template=template)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,chain_type_kwargs={"prompt": prompt})

    answer = chain({"query":q,"context":retriever})
    return answer

#------------------------------------------------------------------------------------------------------------------

#=================
# Background Image
#=================

page_bg_img = '''
<style>
.stApp  {
background-image: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), url("https://img.freepik.com/free-vector/realistic-style-technology-particle-background_23-2148426704.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

#=================
# Streamlit process : Clear history, import OpenAI key, add logo image, upload the files
#=================

# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":

    # loading the OpenAI api key from .env
    load_dotenv(find_dotenv(), override=True)

    # Title and logo
    try :
        image_url = "logo-new.png"
        st.sidebar.image(image_url, caption="", use_column_width=True)
    except :   
        image_url = "ai_logo.png"
        st.sidebar.image(image_url, caption="", use_column_width=True)
    st.subheader("Chat With Your Documents - AI Chatbot")
    with st.sidebar:
        
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number widget
        chunk_size = 1000

        # k number input widget
        k = 3

        # add data button widget
        add_data = st.button('Load Data', on_click=clear_history)

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    # user's question text input widget
    q = st.text_input('Ask a question about the content of your file:')
    
    if q: # if the user entered a question and hit enter
        if 'vs' in st.session_state: # if there's a vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            
            answer = ask_and_get_answer(vector_store, q, k)
    
            # text area widget for the LLM answer
            st.text_area('LLM Answer: ', value=answer["result"], height=400)
            
            # Audio area widget for the TTS part
            sound_file = BytesIO()
            client = OpenAI()
            response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=answer["result"])
            response.stream_to_file("output.mp3")
            st.audio("output.mp3")
            st.divider()
    
            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''
    
            # the current question and answer
            value = f'Q: {q} \nA: {answer["result"]}'
    
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history
    
            # text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history', height=400)
