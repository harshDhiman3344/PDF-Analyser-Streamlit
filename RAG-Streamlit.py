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


# Initialize theme in session state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Theme toggle function
def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# Apply theme CSS based on current theme
if st.session_state.theme == "dark":
    st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        .main-box {
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 10px;
        }
        .stTextInput > div > div > input {
            background-color: #2d2d2d;
            color: white;
        }
        p, div, span, label {
            color: #fafafa !important;
        }
        section[data-testid="stFileUploader"] {
            background-color: #2b5a9e;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #4a7fc1;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp {
            background-color: #ffffff;
            color: #000000;
        }
        .main-box {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
        }
        .stTextInput > div > div > input {
            background-color: #ffffff;
            color: #000000;
        }
        p, div, span, label, h1, h2, h3 {
            color: #000000 !important;
        }
        section[data-testid="stFileUploader"] {
            background-color: #b3d9ff;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #6bb3ff;
        }
    </style>
    """, unsafe_allow_html=True)





#uses user's api key here: 

st.sidebar.title("Settings")

# Theme toggle button in sidebar
theme_button_label = "☀️ Light Mode" if st.session_state.theme == "dark" else "🌙 Dark Mode"
st.sidebar.button(theme_button_label, on_click=toggle_theme)

user_api_key = st.sidebar.text_input("Gemini API Key", type="password")

if user_api_key:
    os.environ["GOOGLE_API_KEY"] = user_api_key
    st.sidebar.success("API key set!")
else:
    st.sidebar.warning("Please enter your Gemini API key to continue.")
    st.stop()  # Stop app until API key is entered



Settings.llm = Gemini(model="gemini-2.5-flash")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


st.markdown('<div class="main-box">', unsafe_allow_html=True)

#Title styles
title_color = "white" if st.session_state.theme == "dark" else "#1e1e1e"
subtitle_color = "#aac4ff" if st.session_state.theme == "dark" else "#5a7fb8"
st.markdown(f"""
<h1 style='text-align:center; color:{title_color}; margin-bottom:0;'>VectorBrain</h1>
<h3 style='text-align:center; color:{subtitle_color}; margin-top:5px;'>RAG PDF Analyzer</h3>
""", unsafe_allow_html=True)

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




st.markdown('</div>', unsafe_allow_html=True)



#VERSION 1 COMPLETE

