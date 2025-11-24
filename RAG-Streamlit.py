import os
import sys
import logging
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini


#loading the env
load_dotenv()


#LOCAL EMBEDDING KARUNGA AAAAA
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


#Logging Setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



# --------------------------
# CUSTOM GEMINI API KEY INPUT
# --------------------------
st.sidebar.title("Settings")
user_api_key = st.sidebar.text_input("Gemini API Key", type="password")

if user_api_key:
    os.environ["GOOGLE_API_KEY"] = user_api_key
    st.sidebar.success("API key set!")
else:
    st.sidebar.warning("Please enter your Gemini API key to continue.")
    st.stop()  # Stop app until API key is entered



Settings.llm = Gemini(model="gemini-2.5-flash")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


st.title("RAG PDF ANALYSER")
st.write("Upload a PDF and ask questions about it.")

uploaded_files = st.file_uploader(
    "Upload your PDFs here",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    data_dir = "uploaded_data"

    # Create folder if not already present
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save uploaded PDFs to disk
    for file in uploaded_files:
        with open(os.path.join(data_dir, file.name), "wb") as f:
            f.write(file.getbuffer())

    st.success("Files uploaded successfully!")


    if st.button("Load and Index"):
        with st.spinner("Working on it"):

            documents = SimpleDirectoryReader(data_dir).load_data()
            print("Docs Loaded Len: ",len(documents))
            print('Now Indexing')
            index = VectorStoreIndex.from_documents(documents, show_progress=True)
    
        st.session_state["index"] = index
        st.success("Indexed Successfully")




if "index" in st.session_state:
    st.write("Ask about the documents")
    prompt = st.text_input("your question:")


    if st.button("Ask"):
        with st.spinner('Thinking'):  
            query_engine = st.session_state["index"].as_query_engine()
            response = query_engine.query(prompt)

        st.write("Response: ")
        st.write(str(response))
    



else:
    st.info("Upload a pdf file to start")
