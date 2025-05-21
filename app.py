import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from io import BytesIO

# Load the saved model
model = load_model('best_model_revised_98.h5') # Replace 'best_model.keras' with the actual filename

st.title("Muhun manga - LAI25-SM013")

option = st.radio("Pilih metode input gambar:", ("Upload File", "Kamera", "Link Gambar"), horizontal=True)

image = None

if option == "Upload File":
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif option == "Kamera":
    camera_file = st.camera_input("Ambil gambar dengan kamera")
    if camera_file is not None:
        image = Image.open(camera_file)
elif option == "Link Gambar":
    url = st.text_input("Masukkan URL gambar")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error("Gagal mengambil gambar dari URL.")

if image is not None:
    # Resize gambar agar tinggi maksimal 250px, lebar menyesuaikan proporsi
    max_height = 250
    w, h = image.size
    if h > max_height:
        new_width = int(w * max_height / h)
        image = image.resize((new_width, max_height))

    image_resized = image.resize((150, 150))
    image_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    image_array = tf.expand_dims(image_array, 0)
    image_array = image_array / 255.0

    predictions = model.predict(image_array)
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mildew']

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Gambar yang dipilih", use_container_width=False)
        
    with col2:
        st.write("**Probabilities:**")
        prob_dict = {name: round(float(pred) * 100, 2) for name, pred in zip(class_names, predictions[0])}
        st.write({k: f"{v}%" for k, v in prob_dict.items()})
    #grafik bar kalo perlu
    #st.bar_chart([round(float(p) * 100, 2) for p in predictions[0]])

    #col1, col2 = st.columns(2)
    disease_descriptions = {
        'Anthracnose': 'Anthracnose adalah penyakit jamur yang menyebabkan bercak gelap pada daun dan buah.',
        'Bacterial Canker': 'Bacterial Canker adalah penyakit bakteri yang menyebabkan luka dan bercak pada batang atau daun.',
        'Cutting Weevil': 'Cutting Weevil adalah serangan kumbang yang memotong bagian tanaman.',
        'Die Back': 'Die Back adalah kondisi di mana bagian ujung tanaman mengering dan mati.',
        'Gall Midge': 'Gall Midge adalah serangan lalat kecil yang menyebabkan pembengkakan pada jaringan tanaman.',
        'Healthy': 'Tanaman dalam kondisi sehat.',
        'Powdery Mildew': 'Powdery Mildew adalah penyakit jamur yang menyebabkan lapisan putih seperti tepung pada permukaan daun.',
        'Sooty Mildew': 'Sooty Mildew adalah jamur yang tumbuh di permukaan daun, biasanya berwarna hitam.'
    }
    disease_solutions = {
        'Anthracnose': 'Buang bagian tanaman yang terinfeksi dan gunakan fungisida sesuai anjuran.',
        'Bacterial Canker': 'Pangkas bagian yang sakit dan gunakan bakterisida. Jaga kebersihan alat pertanian.',
        'Cutting Weevil': 'Kumpulkan dan musnahkan kumbang, gunakan insektisida jika perlu.',
        'Die Back': 'Pangkas bagian yang mati dan perbaiki drainase serta pemupukan.',
        'Gall Midge': 'Buang dan musnahkan jaringan yang bengkak, gunakan insektisida nabati.',
        'Healthy': 'Tidak perlu tindakan, pertahankan perawatan tanaman yang baik.',
        'Powdery Mildew': 'Gunakan fungisida dan tingkatkan sirkulasi udara di sekitar tanaman.',
        'Sooty Mildew': 'Bersihkan permukaan daun dan kendalikan serangga penghasil embun madu.'
    }
    st.write(f"**Predicted class:** {class_names[predicted_class]}")
    with st.expander(f"Apa itu {class_names[predicted_class]}?"):
        st.write(disease_descriptions[class_names[predicted_class]])
    with st.expander(f"Bagaimana mengatasi {class_names[predicted_class]}?"):
        st.write(disease_solutions[class_names[predicted_class]])


