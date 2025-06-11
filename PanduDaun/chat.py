import streamlit as st
import time  

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

st.header("Ingin di Pandu? Mari Ngobrol!")

# Daftar balasan masih manual wak wkwkw
manual_responses = [
    "Halo! Ada yang bisa saya bantu?",
    "Silakan ceritakan masalah Anda.",
    "Anthracnose adalah penyakit jamur yang umum pada tanaman. Gejalanya meliputi bercak hitam pada daun dan buah.",
    "Penyakit bakteri dapat menyebabkan bercak air pada daun dan batang. Pastikan tanaman Anda mendapatkan sirkulasi udara yang baik.",
    "Terima kasih sudah menghubungi kami!"
]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": manual_responses[0]}]
if "response_index" not in st.session_state:
    st.session_state.response_index = 1  # Mulai dari balasan kedua

for message in st.session_state.chat_history:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt := st.chat_input("Kirim pesan"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)
    # Tambahkan delay sebelum balasan
    time.sleep(2.5)  # Delay 2.5 detik
    # Ambil balasan manual berikutnya
    if st.session_state.response_index < len(manual_responses):
        response = manual_responses[st.session_state.response_index]
        st.session_state.response_index += 1
    else:
        response = "Mohon menunggu..."
    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})