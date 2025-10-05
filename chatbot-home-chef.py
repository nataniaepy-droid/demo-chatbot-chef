import streamlit as st
from google import genai
from PIL import Image
import numpy as np
import pypdf
from sklearn.metrics.pairwise import cosine_similarity
import io
import time
import os

# --- 0. Konfigurasi Halaman dan Judul Utama ---
st.set_page_config(
    page_title="Home Chef AI Multimodal RAG",
    layout="centered"
)

st.title("👨‍🍳 Home Chef: Konsultan Resep Cerdas")
st.markdown("Pilih mode di sidebar untuk memulai petualangan memasak Anda!")

# --- 0.1 Inisialisasi Session State (Penting untuk Chatbot & Client) ---
if "messages" not in st.session_state:
    st.session_state.messages = {} 
if "client" not in st.session_state:
    st.session_state.client = None
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = {"chunks": [], "embeddings": None}
if "current_pdf_name" not in st.session_state:
    st.session_state.current_pdf_name = None

# --- 0.2 Fungsi Reset Chat ---
def reset_chat(current_mode):
    """Mengatur ulang riwayat pesan dan sesi chat Gemini."""
    st.session_state.messages[current_mode] = []
    if current_mode == "🍳 Chat Biasa (Pengetahuan Umum)":
        st.session_state.chat_session = None
    st.toast(f"Chat mode {current_mode} telah direset!", icon="🗑️")

# --- 0.3 Fungsi Pemrosesan PDF (RAG Logic) ---
@st.cache_resource(show_spinner="Memproses E-Book Resep (Membaca dan Membuat Embeddings)...")
def process_pdf_and_create_embeddings(pdf_file, _client):
    """Membaca PDF, membaginya menjadi chunks, dan membuat embeddings."""
    
    try:
        # PERBAIKAN: Inisialisasi variabel di awal untuk menghindari NameError
        text_content = ""
        text_chunks = [] 
        
        # 1. Parsing PDF
        reader = pypdf.PdfReader(pdf_file)
        for page in reader.pages:
            # PENTING: Menambahkan spasi agar chunking lebih efektif
            extracted_text = page.extract_text()
            if extracted_text:
                 text_content += extracted_text + "\n\n"
            
        if not text_content.strip():
            st.error("Gagal mengekstrak teks dari PDF. Pastikan isinya adalah teks yang dapat dipilih.")
            return [], None
            
        # 2. Chunking (Pembagian teks)
        CHUNK_SIZE = 2000
        text_chunks = [text_content[i:i + CHUNK_SIZE] for i in range(0, len(text_content), CHUNK_SIZE)]
        
        # 3. Embedding
        st.info(f"Ditemukan {len(text_chunks)} bagian teks. Mulai membuat embeddings...")
        
        embeddings_list = []
        BATCH_SIZE = 50 
        
        for i in range(0, len(text_chunks), BATCH_SIZE):
            batch_chunks = text_chunks[i:i + BATCH_SIZE]
            
            result = _client.models.embed_content( 
                model='text-embedding-004', 
                contents=batch_chunks, 
            )

            # PERBAIKAN: Mengakses .embeddings dan mengekstrak .values untuk float mentah
            for content_embedding in result.embeddings:
                embeddings_list.append(content_embedding.values) 
            
        # Konversi ke NumPy array untuk pencarian cepat
        embeddings_array = np.array(embeddings_list)
        
        st.success("Pemrosesan PDF selesai! E-Book siap digunakan.")
        return text_chunks, embeddings_array
    
    except Exception as e:
        # st.error akan mencetak error jika terjadi
        st.error(f"Terjadi kesalahan pada pemrosesan PDF: {e}")
        return [], None


# --- 1. Sidebar untuk Konfigurasi dan Mode Seleksi ---
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    # Input API Key
    gemini_api_key = st.text_input(
        "Masukkan Gemini API Key Anda:",
        type="password",
        key="gemini_api_input"
    )

    # Inisialisasi Klien Gemini
    if gemini_api_key and st.session_state.client is None:
        try:
            st.session_state.client = genai.Client(api_key=gemini_api_key)
            st.success("Koneksi Gemini berhasil!")
        except Exception as e:
            st.error(f"Gagal menghubungkan ke Gemini: {e}")
            st.session_state.client = None
    elif not gemini_api_key and st.session_state.client is not None:
        st.session_state.client = None

    # Pemilihan Mode
    st.header("✨ Pilih Mode Home Chef")
    mode = st.radio(
        "Pilih Sumber Resep Anda:",
        ("🍳 Chat Biasa (Pengetahuan Umum)", "🖼️ Bicara dengan Gambar (Vision)", "📄 Bicara dengan E-Book (PDF)"),
        key="mode_selection"
    )

    if mode not in st.session_state.messages:
        st.session_state.messages[mode] = []

    # Tombol Reset Chat
    st.button("🔄 Reset Chat", on_click=lambda: reset_chat(mode))

    st.markdown("---")
    st.markdown("Dapatkan [Gemini API Key di Google AI Studio](https://makersuite.google.com/app/apikey).")


# --- 2. Logika Utama: Tampilkan Chat dan Proses Input ---

if st.session_state.client:
    client = st.session_state.client
    
    # Tampilkan riwayat pesan
    for message in st.session_state.messages[mode]:
        with st.chat_message(message["role"]):
            if "image" in message and message["image"] is not None:
                st.image(message["image"], caption="Foto Bahan Masakan Anda", width=250)
            st.markdown(message["content"])

    
    # --- Mode 1: Bicara dengan Gambar (Vision) ---
    if mode == "🖼️ Bicara dengan Gambar (Vision)":
        
        uploaded_image = st.file_uploader("Unggah **Foto Bahan Masakan** Anda:", type=["jpg", "jpeg", "png"], key="vision_uploader")
        
        if uploaded_image:
            st.image(Image.open(uploaded_image), caption="Gambar Anda", width=250)

        prompt = st.chat_input("Apa yang Anda ingin masak dari foto ini?")

        if prompt and uploaded_image:
            with st.chat_message("user"):
                image_to_display = Image.open(uploaded_image)
                st.image(image_to_display, caption="Foto Bahan Masakan Anda", width=250)
                st.markdown(prompt)
                
            st.session_state.messages[mode].append({"role": "user", "content": prompt, "image": Image.open(uploaded_image)})
            
            with st.chat_message("assistant"):
                with st.spinner("Home Chef AI sedang menganalisis bahan di foto..."):
                    try:
                        vision_prompt = f"Berdasarkan gambar ini dan bahan-bahan yang terlihat, berikan saya resep masakan lengkap dalam Bahasa Indonesia. Instruksi pengguna: {prompt}"

                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=[vision_prompt, Image.open(uploaded_image)]
                        )
                        st.markdown(response.text)
                        st.session_state.messages[mode].append({"role": "assistant", "content": response.text})

                    except Exception as e:
                        error_msg = f"Terjadi kesalahan saat pemrosesan gambar: {e}"
                        st.error(error_msg)
                        st.session_state.messages[mode].append({"role": "assistant", "content": error_msg})


    # --- Mode 2: Bicara dengan PDF (RAG) ---
    elif mode == "📄 Bicara dengan E-Book (PDF)":
        
        uploaded_pdf = st.file_uploader("Unggah **File Resep PDF** Anda:", type="pdf", key="pdf_uploader")
        
        # 1. Pemrosesan PDF saat file baru diunggah
        if uploaded_pdf and (uploaded_pdf.name != st.session_state.current_pdf_name):
            st.session_state.pdf_data["chunks"], st.session_state.pdf_data["embeddings"] = \
                process_pdf_and_create_embeddings(uploaded_pdf, client) 
            st.session_state.current_pdf_name = uploaded_pdf.name
            st.session_state.messages[mode] = [{"role": "assistant", "content": f"E-Book '{uploaded_pdf.name}' telah diolah! Sekarang, tanyakan resep atau bahan apa yang harus Anda gunakan."}]
        
        
        # 2. Logika Chat RAG
        if st.session_state.pdf_data["embeddings"] is not None:
            
            if not st.session_state.messages[mode]:
                st.session_state.messages[mode].append({"role": "assistant", "content": f"E-Book '{st.session_state.current_pdf_name}' telah diolah! Sekarang, tanyakan resep atau bahan apa yang harus Anda gunakan."})
            
            prompt = st.chat_input("Tanyakan resep dari e-book Anda...")

            if prompt:
                st.session_state.messages[mode].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Retrieval (Pencarian Vektor)
                with st.spinner("Mencari resep yang relevan di E-Book Anda..."):
                    
                    # Embed query pengguna
                    query_embedding_result = client.models.embed_content(
                        model='text-embedding-004',
                        contents=[prompt], 
                    )
                    
                    # PERBAIKAN: Memaksa array 2D (1, 768) sejak awal untuk cosine_similarity
                    query_embedding = np.array([query_embedding_result.embeddings[0].values]) 

                    # Hitung Cosine Similarity
                    similarities = cosine_similarity(query_embedding, st.session_state.pdf_data["embeddings"])
                    
                    # Ambil 3 indeks teratas (resep paling relevan)
                    top_indices = np.argsort(similarities[0])[::-1][:3]
                    
                    # Susun Konteks
                    retrieved_chunks = [st.session_state.pdf_data["chunks"][i] for i in top_indices]
                    context = "\n\n--KONTEKS RESEP--\n\n" + "\n\n".join(retrieved_chunks)
                
                # Generation (Gemini menjawab dengan Konteks)
                with st.chat_message("assistant"):
                    with st.spinner("Merumuskan jawaban berdasarkan resep E-Book..."):
                        
                        rag_prompt = f"""
                        Anda adalah asisten resep khusus yang hanya menjawab berdasarkan KONTEKS RESEP yang disediakan.
                        Jawab pertanyaan pengguna di bawah ini menggunakan informasi dari konteks tersebut. 
                        Jika konteks tidak mengandung resep yang relevan, katakan saja 'Maaf, saya tidak menemukan resep yang relevan di E-book Anda.'

                        KONTEKS RESEP:
                        ---
                        {context}
                        ---

                        PERTANYAAN PENGGUNA: {prompt}
                        """
                        
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=rag_prompt
                        )
                        st.markdown(response.text)
                        st.session_state.messages[mode].append({"role": "assistant", "content": response.text})

        elif uploaded_pdf is None:
            st.info("Unggah **E-book resep** Anda (format PDF) di atas untuk mengaktifkan mode Bicara dengan E-book.")


    # --- Mode 3: Chat Biasa (Pengetahuan Umum) ---
    elif mode == "🍳 Chat Biasa (Pengetahuan Umum)":
        
        def get_chat_session(client):
            if st.session_state.chat_session is None:
                system_instruction = (
                    "Anda adalah Asisten Resep Profesional yang ramah dan membantu. "
                    "Tugas Anda adalah memandu pengguna dalam mencari resep masakan dalam Bahasa Indonesia "
                    "berdasarkan bahan yang mereka miliki dari pengetahuan umum Anda. "
                    "Jaga percakapan agar tetap menyenangkan. "
                    "Ketika diminta resep, berikan resep lengkap dengan struktur: "
                    "1. Judul Resep, 2. Deskripsi Singkat, 3. Bahan-Bahan Diperlukan, dan 4. Langkah-Langkah Memasak. "
                    "Selalu gunakan format Markdown yang rapi."
                )
                st.session_state.chat_session = client.chats.create(
                    model='gemini-2.5-flash',
                    config=genai.types.GenerateContentConfig(system_instruction=system_instruction)
                )
                st.session_state.messages[mode].append({"role": "assistant", "content": "Hai! Masukkan bahan-bahan apa saja yang Anda miliki. Saya akan bantu carikan resep terbaik dari pengetahuan kuliner saya!"})
            return st.session_state.chat_session

        chat = get_chat_session(client)

        if prompt := st.chat_input("Saya punya ... (masukkan bahan atau pertanyaan resep Anda)"):
            
            st.session_state.messages[mode].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Home Chef AI sedang meracik resep..."):
                    try:
                        response = chat.send_message(prompt)
                        st.markdown(response.text)
                        st.session_state.messages[mode].append({"role": "assistant", "content": response.text})
                    
                    except Exception as e:
                        error_message = f"Terjadi kesalahan: {e}. Coba reset chat atau cek API Key Anda."
                        st.error(error_message)
                        st.session_state.messages[mode].append({"role": "assistant", "content": error_message})

else:
    st.info("Masukkan **Gemini API Key** Anda di sidebar **Konfigurasi** untuk mengaktifkan chatbot.")