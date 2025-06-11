import streamlit as st

st.markdown("""
    <style>
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
    }
    .card {
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(60,60,60,0.07);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        opacity: 0.7;
    }
            </style>
""", unsafe_allow_html=True)

st.header("Panduan Penggunaan Aplikasi Deteksi Penyakit Daun Mangga")

cols = st.columns(3)

st.container(border=True).video("https://s3-us-west-2.amazonaws.com/assets.streamlit.io/videos/hero-video.mp4", autoplay=True)

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