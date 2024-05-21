
#=================
# Import Libraries
#=================

import streamlit as st
import pandas as pd
import os
# from statsmodels.tsa.arima_model import ARIMA
# import statsmodels.api as sm
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
# from langchain.llms import OpenAI
from openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.embeddings import HuggingFaceEmbeddings
from io import BytesIO
from uuid import uuid4

#=================
# Background Image , Chatbot Title and Logo
#=================

page_bg_img = '''
<style>
.stApp  {
background-image: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), url("https://imageio.forbes.com/specials-images/imageserve/6271151bcd7b0b7ffd1fa4e2/Artificial-intelligence-robot/960x0.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)


st.title("Gen AI RAG Chatbot")
try :
    image_url = "logo-new.png"
    st.sidebar.image(image_url, caption="", use_column_width=True)
except :   
    image_url = "https://static.vecteezy.com/system/resources/previews/010/794/341/non_2x/purple-artificial-intelligence-technology-circuit-file-free-png.png"
    st.sidebar.image(image_url, caption="", use_column_width=True)

#=================
# API Key and Files Upload
#=================
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
os.environ["OPENAI_API_KEY"] = openai_api_key


file_format = st.sidebar.selectbox("Select File Format", ["CSV", "PDF", "TXT"])
if file_format == "TXT" :
    file_format = "plain"

uploaded_files = st.sidebar.file_uploader("Upload a file", type=["csv", "txt", "pdf"], accept_multiple_files=True)


def validateFormat(file_format,uploaded_files) :
    """
    Checks if the format of uploaded files matches the selected format.

    Args:
        file_format (str): The selected file format (CSV, PDF, TXT).
        uploaded_files (list): A list of uploaded files.

    Returns:
        bool: True if all file formats match, False otherwise.
    """
    for file in uploaded_files :
        if str(file_format).lower() not in str(file.type).lower():
          return False
    return True


def selectPDFAnalysis() :
    """
    Prompts the user to select the type of PDF analysis (Compare or Merge).

    Returns:
        str: The selected analysis type ("Compare" or "Merge").
    """
    type_pdf = st.selectbox("Select Anaylsis Type on PDFs", ["Compare","Merge"])

    if type_pdf=="Compare" :
       st.write("Analysis Comparing PDFs")
       return "Compare"
    else :
       st.write("Analysis Merging PDFs")  
       return "Merge"

def save_uploadedfile(uploadedfile):
     """
    Saves the uploaded file to the current directory.

    Args:
        uploaded_file (streamlit.UploadedFile): The uploaded file object.

    Returns:
        str: A success message indicating the file has been saved.
     """
     with open(os.path.join(uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File")           

#=================
# Tracing in Langsmith
#=================


unique_id = uuid4().hex[0:8]

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"LangSmith - AI Chatbot with Text to Speech - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "${{ secrets.LANGSMITH_KEY }}"


#=================
# Answer Generation Functions Based on Uploaded File Format
#=================



def history_func(answer,q):  
    """
    Creates and manages the chat history in the Streamlit session state.

    Args:
        answer (str): The answer generated by the LLM.
        q (str): The user's question.
    """
    # if there's no chat history in the session state, create it     
    if 'history' not in st.session_state:
        st.session_state.history = ''

    # the current question and answer
    value = f'Q: {q} \nA: {answer}'

    st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
    h = st.session_state.history

    # text area widget for the chat history
    st.text_area(label='Chat History', value=h, key='history', height=400)
    
    
def CSVAnalysis(uploaded_file) : 
    """
    Performs analysis on a CSV file and allows users to ask questions.

    Args:
        uploaded_file (streamlit.UploadedFile): The uploaded CSV file.
    """
    df = pd.read_csv(uploaded_file)
    left_column,right_column = st.columns(2)
    with left_column:
        st.header("Dataframe Head")
        st.write(df.head())
    with right_column:
        st.header("Dataframe Tail")
        st.write(df.tail())
    save_uploadedfile(uploaded_file)
    fileName = uploaded_file.name
    st.write("fileName is " + fileName)
    user_query = st.text_input('Enter your query')
    agent = create_csv_agent(ChatOpenAI(temperature=0),fileName,verbose=True,max_iterations=100)
    
    if st.button("Answer My Question"):
        st.write("Running the query " , user_query)
        response = agent.run(user_query)
        st.text_area('LLM Answer: ', value=response, height=400)
        sound_file = BytesIO()
        client = OpenAI()
        aud = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=response)
        aud.stream_to_file("output.mp3")
        st.audio("output.mp3")
        history_func(response,user_query)
        
def MergePDFAnalysis(uploaded_files) :
    """
    Merges the text content of multiple PDFs and allows users to ask questions.

    Args:
        uploaded_files (list): A list of uploaded PDF files.
    """
    raw_text = ''
    for file in uploaded_files:
        pdf_reader = PdfReader(file)
        temp_text = ''
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                temp_text += text
        raw_text += temp_text        

    # Split the text into smaller chunks for indexing
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,          
    )

    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI      
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",     # Provide the pre-trained model's path
        model_kwargs={'device':'cpu'}, # Pass the model configuration options
        encode_kwargs={'normalize_embeddings': False})      
    docsearch = FAISS.from_texts(texts, embeddings)

    st.subheader("Enter a question:")
    question = st.text_input("Question")

    if st.button("Answer My Question"):
        # Perform question answering
        docs = docsearch.similarity_search(question)
        chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
        answer = chain.run(input_documents=docs, question=question)
        st.subheader("Answer:")
        st.text_area('LLM Answer: ', value=answer, height=400)
        sound_file = BytesIO()
        client = OpenAI()
        aud = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=answer)
        aud.stream_to_file("output.mp3")
        st.audio("output.mp3")
        history_func(answer,question)
        
        
def ComparePDFAnalysis(uploaded_files) :
    """
    Compares the text content of multiple PDFs and allows users to ask questions about each PDF.

    Args:
        uploaded_files (list): A list of uploaded PDF files.
    """
    tools = []
    llm = ChatOpenAI(temperature=0)
    for file in uploaded_files:
        st.write("File name is ", file.name)
        save_uploadedfile(file)
        loader = PyPDFLoader(file.name)
        pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(pages)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",     # Provide the pre-trained model's path
            model_kwargs={'device':'cpu'}, # Pass the model configuration options
            encode_kwargs={'normalize_embeddings': False}) 
        retriever = FAISS.from_documents(docs, embeddings).as_retriever()
        function_name = file.name.replace('.pdf', '').replace(' ', '_')[:64]
        tools.append(Tool(name=function_name,description=f"useful when you want to answer questions about {function_name}",func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever)))

    agent = initialize_agent(
            agent=AgentType.OPENAI_FUNCTIONS,
            tools=tools,
            llm=llm,
            verbose=True,
        )

    question = st.text_input("Question")    
    if st.button("Answer My Question"):
       st.write("Running the query")
       response = agent.run(question)
       st.text_area('LLM Answer: ', value=response, height=400)                
       sound_file = BytesIO()
       client = OpenAI()
       aud = client.audio.speech.create(
       model="tts-1",
       voice="alloy",
       input=response)
       aud.stream_to_file("output.mp3")
       st.audio("output.mp3")
       history_func(response,question)
       
       
        
def TextAnalysis(uploaded_files) :
    """
    Performs analysis on a text file and allows users to ask questions.

    Args:
        uploaded_files (list): A list of uploaded text files.
    """
    raw_text = ''
    for file in uploaded_files: 
        temp_text = file.read().decode("utf-8")
        raw_text += temp_text    

    # Split the text into smaller chunks for indexing
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",     # Provide the pre-trained model's path
        model_kwargs={'device':'cpu'}, # Pass the model configuration options
        encode_kwargs={'normalize_embeddings': False}) 

    docsearch = FAISS.from_texts(texts, embeddings)

    st.subheader("Enter a question:")
    question = st.text_input("Question")

    if st.button("Answer My Question"):
        # Perform question answering
        docs = docsearch.similarity_search(question)
        chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
        answer = chain.run(input_documents=docs, question=question)
        st.subheader("Answer:")
        st.text_area('LLM Answer: ', value=answer, height=400)      
        sound_file = BytesIO()
        client = OpenAI()
        aud = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=answer)
        aud.stream_to_file("output.mp3")
        st.audio("output.mp3")
        history_func(answer,question)

#=================
# Answer Generation
#=================

        
        
if uploaded_files: 
    if validateFormat(file_format,uploaded_files) :
#        st.write("Format is valid")
        if file_format=="CSV" :
            if len(uploaded_files)>1 :
                st.write("Only 1 CSV file can be uploded")
            else :
#                 st.write("CSV Analysis")
                 for file in uploaded_files :
                    CSVAnalysis(file)
        elif file_format == "PDF" :
            if len(uploaded_files) > 1 :
                select = selectPDFAnalysis()
                if(select=="Compare") :
                    ComparePDFAnalysis(uploaded_files)
                else :
                    MergePDFAnalysis(uploaded_files)
            else :
#                st.write(" Single pdf analysis ")
                MergePDFAnalysis(uploaded_files)
        else :
            TextAnalysis(uploaded_files)
 #           st.write(" Text Analysis ")        
    else :
        st.write("Formats are not valid")
        


