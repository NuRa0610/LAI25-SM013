import streamlit as st

st.header("Panduan Penggunaan Aplikasi Deteksi Penyakit Daun Mangga")

cols = st.columns(3)

st.container(border=True).video("https://youtu.be/PAkCbfRzj-M", autoplay=True)

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