import streamlit as st
import numpy as np

def play_scale(rate):
    sample_rate = rate
    duration = 0.5  # Each note duration of 0.5 seconds

    # Frequencies for the notes do, re, mi, fa, so, la, ti, do
    frequencies = [523.25, 493.88, 440.00, 392.00, 349.23, 329.63, 293.66, 261.63]

    # Generate and concatenate the sine waves for each note
    scale = np.concatenate([
        np.sin(np.pi * freq * np.linspace(0, duration, int(sample_rate * duration), False))
        for freq in frequencies
    ])
    return scale

st.header("Panduan Penggunaan Aplikasi Deteksi Penyakit Daun Mangga")

cols = st.columns(3)
#cols[0].image("https://docs.streamlit.io/logo.svg", use_container_width=True, caption="Streamlit logo")
#st.write("Play a scale")
#st.audio(play_scale(44100), sample_rate=44100)
st.container(border=True).video("https://s3-us-west-2.amazonaws.com/assets.streamlit.io/videos/hero-video.mp4", autoplay=True)

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
