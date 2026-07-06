import streamlit as st
import os
import json
import requests
import tempfile

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Tanya Cuacaku", page_icon="🌤️", layout="wide")

# Konfigurasi API
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GEMINI_API_KEY", "")
NAMA_FILE_PDF = "BAHAN AJAR AGENDA 1 OBSERVASI FENOMENA DAN PARAMETER METEOROLOGI PENERBANGAN.pdf"

# Fungsi BMKG
def cek_cuaca_bmkg(kode_wilayah: str) -> str:
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={kode_wilayah}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return json.dumps({"informasi_lokasi": data['data'][0]['lokasi'], "prakiraan_cuaca": data['data'][0]['cuaca'][0][:4]}, indent=2)
    except Exception as e:
        return f"Error BMKG: {str(e)}"

# RAG PDF
@st.cache_resource
def setup_rag_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

database_pengetahuan = setup_rag_pdf(NAMA_FILE_PDF)

# Tools
@tool
def alat_cek_cuaca(kode_wilayah: str) -> str:
    """Gunakan untuk prakiraan cuaca wilayah (adm4)."""
    return cek_cuaca_bmkg(kode_wilayah)

@tool
def alat_baca_pdf(pertanyaan: str) -> str:
    """Gunakan untuk teori meteorologi."""
    hasil = database_pengetahuan.similarity_search(pertanyaan, k=2)
    return "\n\n".join([doc.page_content for doc in hasil])

# Agent
@st.cache_resource
def buat_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Kamu adalah asisten meteorologi ahli."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    tools = [alat_cek_cuaca, alat_baca_pdf]
    agen = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agen, tools=tools, verbose=True)

agen_cuaca = buat_agent()

# UI
st.title("🌤️ Tanya Cuacaku")
if "messages" not in st.session_state: st.session_state.messages = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if input_user := st.chat_input("Tanyakan cuaca atau teori meteorologi..."):
    st.session_state.messages.append({"role": "user", "content": input_user})
    with st.chat_message("user"): st.markdown(input_user)
    
    with st.chat_message("assistant"):
        respons = agen_cuaca.invoke({"input": input_user, "chat_history": st.session_state.chat_history})
        st.markdown(respons["output"])
    
    st.session_state.messages.append({"role": "assistant", "content": respons["output"]})
    st.session_state.chat_history.append(HumanMessage(content=input_user))
    st.session_state.chat_history.append(AIMessage(content=respons["output"]))
