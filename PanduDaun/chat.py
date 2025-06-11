import streamlit as st
import time  

st.header("Ingin di Pandu? Mari Ngobrol!")

# Daftar balasan masih manual wak wkwkw
manual_responses = [
    "Halo! Ada yang bisa saya bantu?",
    "Silakan ceritakan masalah Anda.",
    "Sepertinya tanaman anda terkena Anthracnose! itu adalah penyakit jamur yang umum pada tanaman. Gejalanya meliputi bercak hitam pada daun dan buah.",
    "Tentu saja ! Berikut adalah deskripsi penyakit Anthracnose:\n\n"
    "Anthracnose adalah penyakit yang disebabkan oleh jamur Colletotrichum spp. \n"
    "Gejalanya berupa bercak coklat kehitaman pada daun, batang, atau buah, \n"
    "yang lama-kelamaan membesar dan menyebabkan jaringan tanaman membusuk. \n"
    "Penyakit ini berkembang pesat pada kondisi lembab dan sering menyerang saat musim hujan.",
    "Untuk mengatasi Anthracnose, Anda dapat melakukan beberapa langkah berikut:\n\n"
    "1. **Pengendalian Kultural**: Jaga kebersihan area tanam, buang daun atau buah yang terinfeksi, dan hindari penanaman tanaman yang rentan di dekat tanaman yang sudah terinfeksi.\n"
    "2. **Pengendalian Hayati**: Gunakan jamur antagonis seperti Trichoderma spp. untuk mengendalikan patogen penyebab Anthracnose.\n"
    "3. **Pengendalian Kimia**: Jika serangan parah, Anda dapat menggunakan fungisida yang sesuai.\n"
    "4. **Pencegahan**: Pilih varietas tanaman yang tahan terhadap Anthracnose, lakukan rotasi tanaman, dan hindari penyiraman berlebihan yang dapat meningkatkan kelembaban.",
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
