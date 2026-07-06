"""
CUACAKU — Asisten Meteorologi & Prakiraan Cuaca
=================================================
Aplikasi Streamlit final, siap deploy ke Streamlit Community Cloud.
Sumber data: BMKG (real-time) + RAG dari dokumen bahan ajar meteorologi.
"""

import streamlit as st
import time
import os
import json
import requests
import tempfile

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Cuacaku", page_icon="🌤️", layout="wide")

# ------------------------------------------
# KONFIGURASI — pakai st.secrets, BUKAN google.colab.userdata
# ------------------------------------------
# API key diambil dari Streamlit Cloud Secrets (Settings > Secrets di dashboard),
# bukan dari Colab Secrets. Format penyimpanan di Streamlit Cloud:
#   GEMINI_API_KEY = "isi_api_key_kamu"
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
    .badge-cuaca {
        background-color: #E6F1FB; color: #0C447C;
        padding: 4px 10px; border-radius: 8px; font-size: 12px;
        display: inline-flex; align-items: center; gap: 4px;
    }
    .badge-info {
        background-color: #EAF3DE; color: #27500A;
        padding: 4px 10px; border-radius: 8px; font-size: 12px;
        display: inline-flex; align-items: center; gap: 4px;
    }
    .subtitle-text {
        color: #6B6B6B; font-size: 13px; margin-top: 0px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# FASE 2 — Fungsi BMKG
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
# FASE 3 — RAG (di-cache supaya tidak rebuild tiap rerun)
# ------------------------------------------
@st.cache_resource(show_spinner="Membangun basis pengetahuan dari PDF...")
def setup_rag_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    dokumen = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(dokumen)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

database_pengetahuan = setup_rag_pdf(NAMA_FILE_PDF)

# ------------------------------------------
# FASE 4 — Tools & Agent
# ------------------------------------------
@tool
def alat_cek_cuaca(kode_wilayah: str) -> str:
    """
    Gunakan alat ini HANYA JIKA pengguna menanyakan prakiraan cuaca atau kondisi cuaca harian di suatu wilayah.
    Input harus berupa kode wilayah tingkat IV (adm4), contoh: '31.71.03.1001'.
    """
    return cek_cuaca_bmkg(kode_wilayah)

@tool
def alat_baca_pdf(pertanyaan: str) -> str:
    """
    Gunakan alat ini JIKA pengguna menanyakan teori, regulasi, definisi, atau observasi
    fenomena meteorologi yang bersumber dari dokumen bahan ajar.
    """
    hasil = database_pengetahuan.similarity_search(pertanyaan, k=2)
    return "\n\n".join(
        f"[Halaman {doc.metadata.get('page', '?')}] {doc.page_content}" for doc in hasil
    )

daftar_alat = [alat_cek_cuaca, alat_baca_pdf]

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Kamu adalah Cuacaku, asisten meteorologi yang ramah dan informatif. "
     "Kamu bisa menjawab soal cuaca real-time maupun teori/pengetahuan meteorologi umum. "
     "Jika pertanyaan tentang cuaca aktual/prakiraan, gunakan alat_cek_cuaca. "
     "Jika pertanyaan tentang teori/regulasi/definisi/awan/fenomena atmosfer, gunakan alat_baca_pdf. "
     "Perhatikan riwayat percakapan sebelumnya untuk memahami konteks pertanyaan lanjutan."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

@st.cache_resource(show_spinner="Menyiapkan AI Agent...")
def buat_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.1)
    agen = create_tool_calling_agent(llm, daftar_alat, prompt)
    return AgentExecutor(agent=agen, tools=daftar_alat, verbose=True, return_intermediate_steps=True)

agen_cuaca = buat_agent()

def ekstrak_teks_jawaban(output_agent) -> str:
    if isinstance(output_agent, str):
        return output_agent
    if isinstance(output_agent, list):
        return "\n".join(
            block.get("text", "") for block in output_agent
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(output_agent)

# ------------------------------------------
# HEADER — logo STMKG + BMKG
# ------------------------------------------
col_logo1, col_logo2, col_title = st.columns([1, 1, 8])
with col_logo1:
    try:
        st.image("Lambang-Logo-STMKG.jpg", width=55)
    except Exception:
        st.markdown("`[Logo STMKG]`")
with col_logo2:
    try:
        st.image("Logo-BMKG png.png", width=55)
    except Exception:
        st.markdown("`[Logo BMKG]`")
with col_title:
    st.markdown("### 🌤️ Cuacaku")
    st.markdown(
        '<p class="subtitle-text">Membantu memberi informasi meteorologi dan '
        'prakiraan cuacamu! (sumber data BMKG)</p>',
        unsafe_allow_html=True
    )

st.markdown(
    '<span class="badge-cuaca">☁️ Info Prakiraan Cuaca</span> &nbsp; '
    '<span class="badge-info">📖 Belajar Meteorologi yuk!</span>',
    unsafe_allow_html=True
)
st.divider()

# ------------------------------------------
# SIDEBAR
# ------------------------------------------
with st.sidebar:
    st.markdown("#### Cuacaku")

    if st.button("➕ Chat baru", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    uploaded_pdf = st.file_uploader("📎 Upload dokumen tambahan (PDF)", type="pdf")
    if uploaded_pdf and uploaded_pdf.name != st.session_state.get("pdf_terakhir_diproses"):
        with st.spinner(f"Mempelajari '{uploaded_pdf.name}'..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name

            loader_baru = PyPDFLoader(tmp_path)
            dokumen_baru = loader_baru.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks_baru = splitter.split_documents(dokumen_baru)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db_baru = FAISS.from_documents(chunks_baru, embeddings)

            database_pengetahuan.merge_from(db_baru)
            st.session_state.pdf_terakhir_diproses = uploaded_pdf.name

        st.success(f"'{uploaded_pdf.name}' berhasil dipelajari dan digabung ke basis pengetahuan!")

    st.markdown("---")
    st.markdown("**Riwayat**")
    for m in st.session_state.get("messages", [])[-6:]:
        if m["role"] == "user":
            st.markdown(f"- {m['content'][:40]}")

    st.markdown("---")
    st.caption("👤 Rafi")

# ------------------------------------------
# STATE
# ------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------------------------
# CHIP SARAN PERTANYAAN
# ------------------------------------------
contoh_pertanyaan = None
if not st.session_state.messages:
    st.markdown("**Coba tanyakan:**")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("☀️ Cuaca hari ini di Kemayoran"):
            contoh_pertanyaan = "Cuaca hari ini di Kemayoran (kode wilayah 31.71.03.1001)"
    with c2:
        if st.button("💨 Apa itu wind shear?"):
            contoh_pertanyaan = "Apa itu wind shear?"
    with c3:
        if st.button("🌩️ Jenis-jenis awan konvektif"):
            contoh_pertanyaan = "Apa saja jenis-jenis awan konvektif?"

# ------------------------------------------
# RENDER RIWAYAT CHAT
# ------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("steps"):
            with st.expander("🔧 Lihat proses tool-calling"):
                for i, (aksi, hasil_tool) in enumerate(msg["steps"], 1):
                    st.markdown(f"**Langkah {i}: `{aksi.tool}`**")
                    st.code(json.dumps(aksi.tool_input, ensure_ascii=False), language="json")
        st.markdown(msg["content"])

# ------------------------------------------
# INPUT USER — panggil agent asli
# ------------------------------------------
input_user = st.chat_input("Tanyakan cuaca atau info meteorologi...")
pesan_final = input_user or contoh_pertanyaan

if pesan_final:
    st.session_state.messages.append({"role": "user", "content": pesan_final})
    with st.chat_message("user"):
        st.markdown(pesan_final)

    with st.chat_message("assistant"):
        with st.spinner("Cuacaku sedang berpikir..."):
            respons = agen_cuaca.invoke({
                "input": pesan_final,
                "chat_history": st.session_state.chat_history
            })
        jawaban = ekstrak_teks_jawaban(respons["output"])
        steps = respons.get("intermediate_steps", [])

        if steps:
            with st.expander("🔧 Lihat proses tool-calling", expanded=True):
                for i, (aksi, hasil_tool) in enumerate(steps, 1):
                    st.markdown(f"**Langkah {i}: `{aksi.tool}`**")
                    st.code(json.dumps(aksi.tool_input, ensure_ascii=False), language="json")
        st.markdown(jawaban)

    st.session_state.messages.append({"role": "assistant", "content": jawaban, "steps": steps})
    st.session_state.chat_history.append(HumanMessage(content=pesan_final))
    st.session_state.chat_history.append(AIMessage(content=jawaban))
    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history = st.session_state.chat_history[-10:]

    st.rerun()
