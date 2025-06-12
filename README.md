# Pandu Daun - Klasifikasi Penyakit Daun pada Tanaman Mangga (Mangifera indica)

Capstone Project Tim LAI25-SM013

## Anggota Tim
- Nu'man  
- Dewi  
- Indira  
- Nida  

## Latar Belakang
Mangga (*Mangifera indica*) adalah komoditas unggulan Indonesia dengan nilai ekonomi tinggi. Namun, produktivitasnya kerap menurun akibat serangan berbagai penyakit daun seperti antraknosa, embun tepung, dan jamur jelaga. Gejala penyakit yang mirip sering kali menyulitkan petani dalam mengidentifikasi secara akurat, yang berujung pada penanganan tidak tepat, kerugian hasil panen, dan pemborosan biaya.

Metode identifikasi konvensional membutuhkan tenaga ahli, biaya mahal, dan tidak selalu tersedia di daerah terpencil. Oleh karena itu, proyek ini bertujuan mengembangkan sistem klasifikasi otomatis berbasis machine learning untuk mendeteksi penyakit daun mangga secara cepat dan akurat melalui citra digital. Model dibangun menggunakan **Convolutional Neural Network (CNN)** dengan framework **TensorFlow** serta data citra dari **Kaggle**.

Untuk mempermudah akses, kami membangun antarmuka berbasis **Streamlit** agar sistem dapat digunakan langsung oleh petani atau pengguna umum. Dengan pendekatan interdisipliner dari anggota tim berlatar belakang **biologi, fisika, dan informatika**, kami menghadirkan solusi praktis bagi pertanian presisi. Proyek ini diharapkan menjadi kontribusi nyata dalam meningkatkan ketahanan pangan, mengurangi kerugian petani, dan mempercepat transformasi digital sektor pertanian Indonesia.

## Link Aplikasi
[Akses Pandu Daun di sini](https://pandu-daun.streamlit.app/)

## Link Repository GitHub
[LAI25-SM013 Repository](https://github.com/NuRa0610/LAI25-SM013)

## Link Preview YouTube
[Tonton preview Pandu Daun di YouTube](https://youtu.be/PAkCbfRzj-M)

## Cara Penggunaan

### Penggunaan Online
1. **Buka aplikasi** melalui link di atas.
2. **Navigasikan ke laman prediksi**.
3. **Unggah gambar** atau **ambil gambar** langsung melalui aplikasi.
4. **Proses analisis akan berjalan secara otomatis**, dan hasil prediksi akan ditampilkan.

### Penggunaan Lokal
Jika ingin menjalankan aplikasi secara lokal, ikuti langkah berikut:

1. **Clone repository dari GitHub**
   ```bash
   git clone https://github.com/NuRa0610/LAI25-SM013
   cd LAI25-SM013

2. **Buat environmen**    
    ```bash
    python -m venv env
    source env/bin/activate  # Untuk MacOS/Linux
    env\Scripts\activate  # Untuk Windows

3. **Instal dependensi**    
    ```bash
    pip install pipenv
    pipenv install

4. **Jalankan streamlit**    
    ```bash
    streamlit run streamlit_app.py

