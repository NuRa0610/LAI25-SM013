import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from io import BytesIO

# model loading
model = load_model('best_model_revised_98.h5')

st.title("PanduDaun - LAI25-SM013")

# image load and process
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
        
    # persentase 
    #with col2:
    #    st.write("**Probabilities:**")
    #    prob_dict = {name: round(float(pred) * 100, 2) for name, pred in zip(class_names, predictions[0])}
    #    st.write({k: f"{v}%" for k, v in prob_dict.items()})
    
    disease_descriptions = {
        'Anthracnose': (
            "Anthracnose adalah penyakit yang disebabkan oleh jamur Colletotrichum spp. "
            "Gejalanya berupa bercak coklat kehitaman pada daun, batang, atau buah, "
            "yang lama-kelamaan membesar dan menyebabkan jaringan tanaman membusuk. "
            "Penyakit ini berkembang pesat pada kondisi lembab dan sering menyerang saat musim hujan."
        ),
        'Bacterial Canker': (
            "Bacterial Canker disebabkan oleh bakteri patogen seperti Xanthomonas atau Pseudomonas. "
            "Gejalanya berupa luka (kanser) pada batang, cabang, atau daun, seringkali disertai eksudat lengket. "
            "Daun bisa menguning, layu, dan rontok. Penyakit ini menyebar melalui percikan air dan alat pertanian yang terkontaminasi."
        ),
        'Cutting Weevil': (
            "Cutting Weevil adalah hama berupa kumbang kecil yang memotong atau melubangi bagian tanaman, "
            "terutama batang muda dan daun. Serangan berat dapat menyebabkan tanaman tumbuh tidak normal, "
            "daun berlubang, dan bahkan kematian tunas muda."
        ),
        'Die Back': (
            "Die Back adalah kondisi di mana ujung cabang atau ranting tanaman mengering dan mati. "
            "Penyebabnya bisa berupa infeksi jamur, bakteri, kekurangan nutrisi, atau kerusakan fisik. "
            "Gejala awal berupa perubahan warna pada ujung daun/cabang, lalu jaringan mengering dan mati ke arah pangkal."
        ),
        'Gall Midge': (
            "Gall Midge adalah serangan lalat kecil (Ordo Cecidomyiidae) yang menyebabkan terbentuknya gall (bengkak) "
            "pada jaringan tanaman. Larva berkembang di dalam gall dan mengganggu pertumbuhan normal tanaman. "
            "Serangan berat dapat menyebabkan deformasi daun dan penurunan hasil."
        ),
        'Healthy': (
            "Tanaman dalam kondisi sehat, tidak menunjukkan gejala penyakit atau serangan hama. "
            "Daun berwarna hijau segar, pertumbuhan normal, dan tidak ada bercak, luka, atau deformasi."
        ),
        'Powdery Mildew': (
            "Powdery Mildew adalah penyakit jamur yang ditandai dengan munculnya lapisan putih seperti tepung "
            "pada permukaan daun, batang, atau bunga. Penyakit ini disebabkan oleh jamur dari famili Erysiphaceae. "
            "Infeksi berat dapat menghambat fotosintesis dan pertumbuhan tanaman."
        ),
        'Sooty Mildew': (
            "Sooty Mildew adalah jamur permukaan yang tumbuh di atas embun madu yang dihasilkan oleh serangga seperti kutu daun. "
            "Daun tampak berwarna hitam seperti berjelaga, menghambat proses fotosintesis, dan menurunkan kualitas hasil panen."
        )
    }
    disease_solutions = {
        'Anthracnose': (
            "1. Buang dan musnahkan bagian tanaman yang terinfeksi untuk mencegah penyebaran.\n"
            "2. Semprotkan fungisida berbahan aktif seperti mankozeb atau klorotalonil sesuai dosis anjuran.\n"
            "3. Jaga kebersihan kebun dan hindari kelembaban berlebih.\n"
            "4. Lakukan rotasi tanaman dan pilih varietas tahan penyakit jika tersedia."
        ),
        'Bacterial Canker': (
            "1. Pangkas dan musnahkan bagian tanaman yang menunjukkan gejala kanser.\n"
            "2. Sterilkan alat pertanian sebelum dan sesudah digunakan.\n"
            "3. Semprotkan bakterisida berbahan aktif tembaga secara berkala.\n"
            "4. Hindari penyiraman berlebihan dan perbaiki drainase lahan."
        ),
        'Cutting Weevil': (
            "1. Kumpulkan dan musnahkan kumbang dewasa secara manual jika populasinya rendah.\n"
            "2. Gunakan insektisida berbahan aktif sesuai rekomendasi jika serangan berat.\n"
            "3. Lakukan sanitasi kebun dan buang sisa tanaman yang menjadi tempat berkembang biak hama.\n"
            "4. Tanam tanaman penutup tanah untuk mengurangi populasi hama."
        ),
        'Die Back': (
            "1. Pangkas bagian tanaman yang mati hingga ke jaringan sehat.\n"
            "2. Oleskan fungisida pada luka bekas potongan untuk mencegah infeksi lanjutan.\n"
            "3. Perbaiki drainase dan hindari genangan air di sekitar tanaman.\n"
            "4. Berikan pupuk seimbang untuk meningkatkan ketahanan tanaman."
        ),
        'Gall Midge': (
            "1. Buang dan musnahkan daun atau jaringan yang membengkak (gall).\n"
            "2. Semprotkan insektisida nabati atau berbahan aktif sesuai anjuran jika serangan berat.\n"
            "3. Lakukan monitoring rutin dan pasang perangkap serangga.\n"
            "4. Jaga kebersihan kebun dan hindari penumpukan sisa tanaman."
        ),
        'Healthy': (
            "1. Lanjutkan perawatan rutin seperti penyiraman, pemupukan, dan pengendalian hama/penyakit secara preventif.\n"
            "2. Pantau kondisi tanaman secara berkala untuk deteksi dini masalah.\n"
            "3. Jaga kebersihan lingkungan sekitar tanaman."
        ),
        'Powdery Mildew': (
            "1. Pangkas dan musnahkan daun yang terinfeksi berat.\n"
            "2. Semprotkan fungisida berbahan aktif sulfur atau miklobutanil sesuai dosis anjuran.\n"
            "3. Tingkatkan sirkulasi udara di sekitar tanaman dengan jarak tanam yang cukup.\n"
            "4. Hindari penyiraman pada malam hari."
        ),
        'Sooty Mildew': (
            "1. Bersihkan permukaan daun dengan air bersih atau kain lembab.\n"
            "2. Kendalikan serangga penghasil embun madu seperti kutu daun dengan insektisida nabati.\n"
            "3. Jaga kebersihan tanaman dan lingkungan sekitar.\n"
            "4. Lakukan pemangkasan cabang yang terlalu rimbun untuk meningkatkan sirkulasi udara."
        )
    }
    st.write(f"**Predicted class:** {class_names[predicted_class]}")
    with st.expander(f"Apa itu {class_names[predicted_class]}?"):
        st.write(disease_descriptions[class_names[predicted_class]])
    with st.expander(f"Bagaimana mengatasi {class_names[predicted_class]}?"):
        st.write(disease_solutions[class_names[predicted_class]])


