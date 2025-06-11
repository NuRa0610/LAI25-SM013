import streamlit as st
from cards import (
    chat_card,
    media_card,
    prediction_card,
)

st.markdown("""
    <style>
    body {
        background-color: #f7faf5;
    }
    .big-title {
        font-size:2.2rem;
        color:#388e3c;
        font-weight:700;
    }
    .card {
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(60,60,60,0.07);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        opacity: 0.7;
    }
    .green-btn {
        background: #4caf50;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        border: none;
        font-weight: 600;
        font-size: 1rem;
        margin-top: 1rem;
        cursor:pointer;
    }
    .stApp {
        background-image:
            url('https://raw.githubusercontent.com/NuRa0610/LAI25-SM013/main/PanduDaun/bg.png'),
            url('https://raw.githubusercontent.com/NuRa0610/LAI25-SM013/main/PanduDaun/bg.png'),
            url('https://raw.githubusercontent.com/NuRa0610/LAI25-SM013/main/PanduDaun/bg.png'),
            url('https://raw.githubusercontent.com/NuRa0610/LAI25-SM013/main/PanduDaun/bg.png');
        background-position:
            10% 20%,
            90% 10%,
            20% 80%,
            80% 60%;
        background-size: 220px 220px, 150px 150px, 180px 180px, 150px 150px;
        background-repeat: no-repeat;
        background-attachment: fixed;
    </style>
""", unsafe_allow_html=True)

st.markdown('''
<div class="card">
    <span class="big-title">Deteksi Penyakit Daun Mangga</span><br>
    <span class="big-title">Secara Otomatis</span><br>
    Unggah foto daun mangga dan temukan solusinya secara otomatis dengan teknologi AI.<br>
    <a href="/app"><button class="green-btn">Mulai Deteksi</button></a>
</div>
''', unsafe_allow_html=True)
#st.page_link("app.py", label="Mulai Deteksi", icon="🔎")

col1, col2 = st.columns([2,1])
with col1:
    st.markdown('<div class="card"><b>PanduDaun</b> adalah aplikasi yang membantu mendeteksi penyakit daun mangga secara cepat dan akurat dengan teknologi AI.<br><br>'
                '<b>Highlight Fitur Utama:</b><br>'
                '✅ Deteksi cepat dan akurat<br>'
                '✅ Saran penanganan berbasis data<br>'
                '✅ Ramah pengguna<br></div>', unsafe_allow_html=True)
with col2:
    st.image("Capstone_logo.png")

st.markdown('<div class="card"><b>Panduan Penggunaan Aplikasi</b></div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.image("https://images.unsplash.com/photo-1502741338009-cac2772e18bc", use_container_width=True)
    st.markdown("**1. Ambil atau Unggah Foto**<br>Gunakan kamera atau unggah gambar daun tomat.", unsafe_allow_html=True)
with col2:
    st.image("https://images.unsplash.com/photo-1464983953574-0892a716854b", use_container_width=True)
    st.markdown("**2. Analisis Otomatis**<br>AI akan menganalisis gambar secara otomatis.", unsafe_allow_html=True)
with col3:
    st.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb", use_container_width=True)
    st.markdown("**3. Lihat Hasil dan Saran**<br>Hasil deteksi dan saran penanganan akan muncul.", unsafe_allow_html=True)

st.markdown('<div class="card"><b>Bantuan lebih lanjut</b></div>', unsafe_allow_html=True)
cols = st.columns(2)

with cols[0].container(height=310):
    chat_card()
with cols[1].container(height=310):
    media_card()

st.markdown('<div class="card"><b>Daftar Penyakit Tomat</b></div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.image("https://images.unsplash.com/photo-1465101046530-73398c7f28ca", use_container_width=True)
    st.markdown("**Anthracnose**<br>Adalah penyakit yang disebabkan oleh jamur dan ditandai dengan bercak-bercak gelap pada daun.", unsafe_allow_html=True)
with col2:
    st.image("https://images.unsplash.com/photo-1464983953574-0892a716854b", use_container_width=True)
    st.markdown("**Bacterial Spot**<br>Bercak hitam pada daun dan buah.", unsafe_allow_html=True)
with col3:
    st.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb", use_container_width=True)
    st.markdown("**Early Blight**<br>Bercak konsentris pada daun.", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="background:#388e3c;color:white;padding:1.5rem;border-radius:10px;margin-top:2rem;">
        <b>PanduDaun</b> &copy; 2025 | Aplikasi AI untuk petani Indonesia
    </div>
""", unsafe_allow_html=True)