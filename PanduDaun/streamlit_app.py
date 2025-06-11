import streamlit as st
import pandas as pd
import numpy as np
from cards import (
    chat_card,
)

# Tambahkan CSS custom di bawah ini
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&family=Space+Mono:wght@400;700&display=swap');
    .stApp {     
        background-color: #D0F0C0 !important;
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
        color: #3d3a2a !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    header[data-testid="stHeader"] {
        background-color: #D0F0C0 !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #B8E6A0 !important;
        color: #3d3a2a !important;
    }
    .st-cq, .st-cp, .st-cq * {
        border-color: #388e3c !important;
        border-radius: 0.6rem !important;
    }
    .stButton>button, button[kind="secondary"] {
        background: #4caf50 !important;
        color: #fff !important;
        border-radius: 0.6rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    .stButton>button:hover, button[kind="secondary"]:hover {
        background: #388e3c !important;
        color: #fff !important;
    }
    a {
        color: #3d3a2a !important;
    }
    code, pre {
        background: powderBlue !important;
        font-family: 'Space Mono', monospace !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    .carousel-card {
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(60,60,60,0.07);
        padding: 1.5rem 1rem 1rem 1rem;
        margin-bottom: 2rem;
        margin-top: 1.5rem;
        opacity: 0.95;
    }
    .carousel-info {
        text-align: center;
        font-weight: 600;
        color: #388e3c;
        margin-bottom: 1rem;
        font-size: 1.1rem;
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
    </style>
""", unsafe_allow_html=True)

if "init" not in st.session_state:
    st.session_state.chart_data = pd.DataFrame(
        np.random.randn(20, 3), columns=["a", "b", "c"]
    )
    st.session_state.map_data = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=["lat", "lon"],
    )
    st.session_state.init = True


pages = [
    st.Page(
        "home.py",
        title="Home",
        icon=":material/home:"
    ),
    
    st.Page(
        "chat.py",
        title="Chat",
        icon=":material/chat:"
    ),

    st.Page(
        "media.py",
        title="Media",
        icon=":material/image:"
    ),

    st.Page(
        "app.py",
        title="Prediksi",
        icon=":material/insert_chart:"
    ),

    st.Page(
        "team.py",
        title="Tentang Kami",
        icon=":material/people:"
    )
]

page = st.navigation(pages)
page.run()

with st.sidebar.container(height=310):
    if page.title == "Chat":
        chat_card()
    elif page.title == "Media":
        st.page_link("media.py", label="Media", icon=":material/image:")
        st.video("https://youtu.be/eAVGPxPghpU", autoplay=True)
    elif page.title == "Prediksi":
        st.page_link("app.py", label="Panduan", icon=":material/insert_chart:")
        st.write("Silahkan upload gambar berdasarkan pilihan yang ada.")
        st.write("Gambar yang masuk akan otomatis terprediksi.")
    elif page.title == "Tentang Kami":
        st.page_link("team.py", label="Tentang Kami", icon=":material/people:")
        st.markdown(
            "Projek ini dikembangkan oleh tim **LAI25-SM013**. Dalam kegiatan Laskar AI. <br>"
            "Untuk informasi lebih lanjut, kunjungi [GitHub](https://github.com/NuRa0610/LAI25-SM013)."
            , unsafe_allow_html=True
        )
    else:
        st.page_link("home.py", label="Home", icon=":material/home:")
        st.write("Selamat datang di Aplikasi Deteksi Penyakit Daun Mangga!")
        st.write(
            "Pilih halaman yang dituju dari navigasi diatas, untuk berpindah ke halaman tersebut."
        )

st.sidebar.caption(
    "Aplikasi ini menggunakan font [Space Grotesk](https://fonts.google.com/specimen/Space+Grotesk) "
    "dan [Space Mono](https://fonts.google.com/specimen/Space+Mono)."
)