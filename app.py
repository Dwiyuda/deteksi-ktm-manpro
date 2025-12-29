import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from transformers import AutoModelForImageClassification
import time

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Sistem Deteksi KTM",
    page_icon="🎓",
    layout="centered"
)

# Judul & Deskripsi
st.markdown("<h1 style='text-align: center;'>Sistem Deteksi Keaslian KTM</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Menggunakan Convolutional Neural Network (EfficientNet-B0)</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 2. LOAD MODEL (Hanya sekali jalan)
# ==========================================
@st.cache_resource
def load_model():
    """
    Fungsi ini memuat model ke memori (RAM) dan di-cache 
    agar tidak perlu loading ulang setiap kali user upload gambar.
    """
    try:
        # Panggil Arsitektur "Wadah" (HuggingFace EfficientNet)
        model = AutoModelForImageClassification.from_pretrained(
            "google/efficientnet-b0", 
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        
        # Masukkan "Otak" (Weights) dari file .pth Anda
        # Pastikan file .pth ada di satu folder dengan app.py
        state_dict = torch.load("best_model_ktm_final.pth", map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval() # Mode ujian (matikan dropout)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Load model di awal
with st.spinner("Sedang memuat model AI... Mohon tunggu sebentar..."):
    model = load_model()

# ==========================================
# 3. PREPROCESSING (Resep Rahasia CenterCrop)
# ==========================================
def preprocess_image(image):
    """
    Mengubah gambar user menjadi format yang dimengerti AI.
    PENTING: Menggunakan CenterCrop agar gambar tidak gepeng.
    """
    # 1. Konversi ke RGB (jaga-jaga file PNG transparan)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 2. Transformasi Standar (Sama persis dengan Training)
    # Ini menggantikan cara lama image.resize((224,224)) yang bikin gepeng
    transform = transforms.Compose([
        transforms.Resize(256),        # Resize sisi terpendek jadi 256
        transforms.CenterCrop(224),    # Potong tengah 224x224 (Fokus ke objek)
        transforms.ToTensor(),         # Ubah jadi angka (Tensor)
        transforms.Normalize(          # Samakan standar warna
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Tambahkan dimensi batch (karena model minta input [1, 3, 224, 224])
    input_tensor = transform(image).unsqueeze(0)
    
    return input_tensor

# ==========================================
# 4. LOGIKA UTAMA (Upload & Prediksi)
# ==========================================

# Tombol Upload
uploaded_file = st.file_uploader("Unggah Foto KTM (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan Gambar User
    image = Image.open(uploaded_file)
    
    # Buat 2 Kolom: Kiri (Gambar), Kanan (Hasil)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Foto yang Diunggah", use_column_width=True)

    # Proses Prediksi
    if model is not None:
        # Preprocessing
        input_tensor = preprocess_image(image)

        # Prediksi
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs.logits, dim=1)
            
            # Ambil Probabilitas (Sesuai temuan: 0=Asli, 1=Palsu)
            prob_asli = probs[0][0].item() * 100
            prob_palsu = probs[0][1].item() * 100
            
            # Keputusan Akhir (Threshold Mayoritas)
            pred_idx = torch.argmax(probs, dim=1).item()
            
        # Tampilkan Hasil di Kolom Kanan
        with col2:
            st.subheader("Hasil Analisis:")
            
            # Progress Bar Sederhana untuk visualisasi keyakinan
            st.write("Skor Keyakinan:")
            if pred_idx == 0:
                st.progress(int(prob_asli))
            else:
                st.progress(int(prob_palsu))
            
            # Logika Tampilan (0=ASLI, 1=PALSU)
            if pred_idx == 0:
                st.success("✅ **KTM TERDETEKSI ASLI**")
                st.metric("Tingkat Keyakinan (Confidence)", f"{prob_asli:.2f}%")
                st.info("Sistem mendeteksi fitur fisik, font, dan hologram yang sesuai dengan standar KTM Asli.")
            else:
                st.error("🚨 **KTM TERDETEKSI PALSU**")
                st.metric("Tingkat Keyakinan (Confidence)", f"{prob_palsu:.2f}%")
                st.warning("Terdeteksi anomali pada tekstur atau pola visual yang mengindikasikan pemalsuan.")

    # Tampilkan Detail Teknis (Opsional - Bagus untuk demo skripsi)
    with st.expander("🔍 Lihat Detail Teknis (Untuk Penguji)"):
        st.write(f"**Probabilitas Kelas 0 (Asli):** {prob_asli:.4f}%")
        st.write(f"**Probabilitas Kelas 1 (Palsu):** {prob_palsu:.4f}%")
        st.write("**Metode Preprocessing:** Resize(256) -> CenterCrop(224) -> Normalize")
        st.write("**Model:** EfficientNet-B0 (Transfer Learning)")