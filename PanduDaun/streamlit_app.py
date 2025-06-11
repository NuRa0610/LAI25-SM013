import streamlit as st
import pandas as pd
import numpy as np
from cards import (
    chat_card,
)

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
]

page = st.navigation(pages)
page.run()

with st.sidebar.container(height=310):
    if page.title == "Chat":
        chat_card()
    elif page.title == "Media":
        st.page_link("media.py", label="Media", icon=":material/image:")
        st.video("https://s3-us-west-2.amazonaws.com/assets.streamlit.io/videos/hero-video.mp4", autoplay=True)
    elif page.title == "Prediksi":
        st.page_link("app.py", label="Panduan", icon=":material/insert_chart:")
        st.write("Silahkan upload gambar berdasarkan pilihan yang ada.")
        st.write("Gambar yang masuk akan otomatis terprediksi.")
    else:
        st.page_link("home.py", label="Home", icon=":material/home:")
        st.write("Selamat datang di Aplikasi Deteksi Penyakit Daun Mangga!")
        st.write(
            "Pilih halaman dari atas. Thumbnail sidebar ini menunjukkan subset "
            "elemen dari setiap halaman sehingga Anda dapat melihat tema sidebar."
        )

st.sidebar.caption(
    "Aplikasi ini menggunakan font [Space Grotesk](https://fonts.google.com/specimen/Space+Grotesk) "
    "dan [Space Mono](https://fonts.google.com/specimen/Space+Mono)."
)