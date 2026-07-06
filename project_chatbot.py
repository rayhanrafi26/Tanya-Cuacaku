"""
CUACAKU — Asisten Meteorologi & Prakiraan Cuaca
=================================================
Aplikasi Streamlit final, siap deploy ke Streamlit Community Cloud.
"""

import streamlit as st
import os
import json
import requests
import tempfile

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Cuacaku", page_icon="🌤️", layout="wide")

# ------------------------------------------
# KONFIGURASI API KEY
# ------------------------------------------
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "⚠️ GEMINI_API_KEY belum di-set. Tambahkan di Streamlit Cloud: "
        "Settings → Secrets, format: GEMINI_API_KEY = \"isi-api-key-mu\""
    )
    st.stop()

NAMA_FILE_PDF = "BAHAN AJAR AGENDA 1 OBSERVASI FENOMENA DAN PARAMETER METEOROLOGI PENERBANGAN.pdf"

# ------------------------------------------
# CUSTOM CSS
# ------------------------------------------
st.markdown("""
<style>
    .badge-cuaca { background-color: #E6F1FB; color: #0C447C; padding: 4px 10px; border-radius: 8px; font-size: 12px; }
    .badge-info { background-color: #EAF3DE; color: #27500A; padding: 4px 10px; border-radius: 8px; font-size: 12px; }
    .subtitle-text { color: #6B6B6B; font-size: 13px; margin-top: 0px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# FUNGSI BMKG
# ------------------------------------------
def cek_cuaca_bmkg(kode_wilayah: str) -> str:
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={kode_wilayah}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        lokasi = data['data'][0]['lokasi']
        prakiraan = data['data'][0]['cuaca'][0][:4]
        return json.dumps({"informasi_lokasi": lokasi, "prakiraan_cuaca": prakiraan}, indent=2)
    except Exception as e:
        return f"Terjadi kesalahan saat mengambil data dari BMKG: {str(e)}"

# ------------------------------------------
# RAG PDF (Basis Pengetahuan)
# ------------------------------------------
@st.cache_resource(show_spinner="Membangun basis pengetahuan dari PDF...")
def setup_rag_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        return None
    loader = PyPDFLoader(pdf_path)
    dokumen = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(dokumen)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

database_pengetahuan = setup_rag_pdf(NAMA_FILE_PDF)

# ------------------------------------------
# TOOLS & AGENT (LANGGRAPH)
# ------------------------------------------
@tool
def alat_cek_cuaca(kode_wilayah: str) -> str:
    """Gunakan HANYA JIKA pengguna menanyakan prakiraan cuaca wilayah (adm4), contoh: '31.71.03.1001'."""
    return cek_cuaca_bmkg(kode_wilayah)

@tool
def alat_baca_pdf(pertanyaan: str) -> str:
    """Gunakan JIKA pengguna menanyakan teori, regulasi, definisi fenomena meteorologi penerbangan."""
    if database_pengetahuan is None: return "File PDF bahan ajar tidak ditemukan di sistem."
    hasil = database_pengetahuan.similarity_search(pertanyaan, k=2)
    return "\n\n".join(f"[Halaman {doc.metadata.get('page', '?')}] {doc.page_content}" for doc in hasil)

@st.cache_resource(show_spinner="Menyiapkan AI Agent...")
def buat_agent():
    # Menggunakan model Flash-Lite agar bebas dari error limit API
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)
    tools = [alat_cek_cuaca, alat_baca_pdf]
    system_prompt = (
        "Kamu adalah Cuacaku, asisten ahli meteorologi. "
        "Jika pertanyaan tentang cuaca aktual/prakiraan, panggil alat_cek_cuaca. "
        "Jika pertanyaan tentang teori/awan/fenomena atmosfer, panggil alat_baca_pdf. "
        "Jawab dengan ramah, rapi, dan mudah dipahami."
    )
    # LangGraph Agent yang jauh lebih stabil
    return create_react_agent(llm, tools, state_modifier=system_prompt)

agen_cuaca = buat_agent()

# ------------------------------------------
# UI: HEADER & SIDEBAR
# ------------------------------------------
col_logo1, col_logo2, col_title = st.columns([1, 1, 8])
with col_logo1:
    st.image("Lambang-Logo-STMKG.jpg", width=55) if os.path.exists("Lambang-Logo-STMKG.jpg") else st.markdown("`[Logo STMKG]`")
with col_logo2:
    st.image("Logo-BMKG png.png", width=55) if os.path.exists("Logo-BMKG png.png") else st.markdown("`[Logo BMKG]`")
with col_title:
    st.markdown("### 🌤️ Cuacaku")
    st.markdown('<p class="subtitle-text">Membantu memberi informasi meteorologi dan prakiraan cuacamu!</p>', unsafe_allow_html=True)

st.markdown('<span class="badge-cuaca">☁️ Cuaca</span> <span class="badge-info">📖 Info umum</span>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("#### Cuacaku")
    if st.button("➕ Chat baru", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    uploaded_pdf = st.file_uploader("📎 Upload dokumen tambahan (PDF)", type="pdf")
    if uploaded_pdf and uploaded_pdf.name != st.session_state.get("pdf_terakhir_diproses"):
        with st.spinner(f"Mempelajari '{uploaded_pdf.name}'..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                db_baru = FAISS.from_documents(
                    RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(PyPDFLoader(tmp.name).load()), 
                    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                )
            if database_pengetahuan: database_pengetahuan.merge_from(db_baru)
            st.session_state.pdf_terakhir_diproses = uploaded_pdf.name
        st.success("Dokumen berhasil digabung!")
    st.caption("👤 Mode Interaktif Aktif")

# ------------------------------------------
# STATE & CHAT LOGIC
# ------------------------------------------
# Menggunakan format pesan asli LangChain agar sinkron dengan memori LangGraph
if "messages" not in st.session_state: 
    st.session_state.messages = []

# Tampilkan riwayat
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"): st.markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"): st.markdown(msg.content)

# Input user
input_user = st.chat_input("Tanyakan cuaca atau info meteorologi...")

if input_user:
    st.session_state.messages.append(HumanMessage(content=input_user))
    with st.chat_message("user"): 
        st.markdown(input_user)

    with st.chat_message("assistant"):
        with st.spinner("Cuacaku sedang berpikir..."):
            try:
                # Agent memproses seluruh riwayat obrolan secara otomatis
                respons = agen_cuaca.invoke({"messages": st.session_state.messages})
                
                # Mengambil respons teks terakhir dari agent
                jawaban_akhir = respons["messages"][-1].content
                st.markdown(jawaban_akhir)
                
                # Menyimpan jawaban ke dalam state memori
                st.session_state.messages.append(AIMessage(content=jawaban_akhir))
            except Exception as e:
                st.error(f"Gagal merespons: {str(e)}")
