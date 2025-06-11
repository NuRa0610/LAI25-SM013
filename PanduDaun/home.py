import streamlit as st
from cards import (
    chat_card,
    media_card,
    prediction_card,
)

st.markdown('''
<div class="card">
    <span class="big-title">Deteksi Penyakit Daun Mangga</span><br>
    <span class="big-title">Secara Otomatis</span><br>
    Unggah foto daun mangga dan temukan solusinya secara otomatis dengan teknologi AI.<br>
    <a href="/app"><button class="green-btn">Mulai Deteksi</button></a>
</div>
''', unsafe_allow_html=True)

col1, col2 = st.columns([2,1])
with col1:
    st.markdown('<div class="card"><b>PanduDaun</b> adalah aplikasi yang membantu mendeteksi penyakit daun mangga secara cepat dan akurat dengan teknologi AI.<br><br>'
                '<b>Highlight Fitur Utama:</b><br>'
                '✅ Deteksi cepat dan akurat<br>'
                '✅ Saran penanganan berbasis data<br>'
                '✅ Ramah pengguna<br></div>', unsafe_allow_html=True)
with col2:
    st.image("https://raw.githubusercontent.com/NuRa0610/LAI25-SM013/main/PanduDaun/Capstone_logo.png")

st.markdown('<div class="card"><b>Panduan Penggunaan Aplikasi</b></div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.image("https://raw.githubusercontent.com/NuRa0610/LAI25-SM013/main/PanduDaun/img_3.png", use_container_width=True)
    st.markdown("**1. Ambil atau Unggah Foto**<br>Gunakan kamera atau unggah gambar daun tomat.", unsafe_allow_html=True)
with col2:
    st.image("https://raw.githubusercontent.com/NuRa0610/LAI25-SM013/main/PanduDaun/img_1.png", use_container_width=True)
    st.markdown("**2. Analisis Otomatis**<br>AI akan menganalisis gambar secara otomatis.", unsafe_allow_html=True)
with col3:
    st.image("https://raw.githubusercontent.com/NuRa0610/LAI25-SM013/main/PanduDaun/img_2.png", use_container_width=True)
    st.markdown("**3. Lihat Hasil dan Saran**<br>Hasil deteksi dan saran penanganan akan muncul.", unsafe_allow_html=True)

st.markdown('<div class="card"><b>Bantuan lebih lanjut</b></div>', unsafe_allow_html=True)
cols = st.columns(2)

with cols[0].container(height=310):
    chat_card()
with cols[1].container(height=310):
    media_card()

st.markdown('<div class="card"><b>Daftar Penyakit Daun Mangga</b></div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

images = [
    {
        "url": "https://raw.githubusercontent.com/NuRa0610/LAI25-SM013/main/PanduDaun/anth.jpg",
        "caption": "Anthracnose (Penyakit Jamur)"
    },
    {
        "url": "https://raw.githubusercontent.com/NuRa0610/LAI25-SM013/main/PanduDaun/bacterial.jpg",
        "caption": "Bacterial Blight (Penyakit Bakteri)"
    },
    {
        "url": "https://raw.githubusercontent.com/NuRa0610/LAI25-SM013/main/PanduDaun/gall.jpg",
        "caption": "Galls (Penyakit Gall)"
    },
    {
        "url": "https://raw.githubusercontent.com/NuRa0610/LAI25-SM013/main/PanduDaun/powder.jpg",
        "caption": "Powdery Mildew (Penyakit Embun Tepung)"
    },
    {
        "url": "https://raw.githubusercontent.com/NuRa0610/LAI25-SM013/main/PanduDaun/sooty.jpg",
        "caption": "Sooty Mold (Penyakit Jamur Hitam)"
    },
]
if "idx" not in st.session_state:
    st.session_state.idx = 0

col_prev, col_info, col_next = st.columns([1, 2, 1])
with col_prev:
    if st.button("⬅️ Sebelumnya"):
        if st.session_state.idx == 0:
            st.session_state.idx = len(images) - 3
        else:
            st.session_state.idx -= 1
with col_next:
    if st.button("Berikutnya ➡️"):
        if st.session_state.idx == len(images) - 3:
            st.session_state.idx = 0
        else:
            st.session_state.idx += 1
with col_info:
    st.markdown(
        f'<div class="carousel-info">Gambar {st.session_state.idx+1} - {st.session_state.idx+3} dari {len(images)}</div>',
        unsafe_allow_html=True
    )

cols = st.columns(3)
for i, col in enumerate(cols):
    col.image(images[st.session_state.idx + i]["url"], use_container_width=True)
    col.caption(
        f'<span style="font-size:1rem;font-weight:500;">{images[st.session_state.idx + i]["caption"]}</span>',
        unsafe_allow_html=True
    )

# Footer
st.markdown("""
    <div style="background:#388e3c;color:white;padding:1.5rem;border-radius:10px;margin-top:2rem;">
        <b>PanduDaun</b> &copy; 2025 | Aplikasi AI untuk petani Indonesia
    </div>
""", unsafe_allow_html=True)