

import streamlit as st
import pandas as pd
import joblib

# Judul aplikasi
st.title('🦉 Asuransi Vehicle Prediction App')
st.write("""
## Prediksi Menggunakan Model Machine Learning
Masukkan data secara manual untuk melakukan prediksi
""")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_rf_pipeline_model.pkl')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

model = load_model()

# Daftar fitur dan konfigurasi input
FEATURES = {
    'Vehicle Class': {
        'type': 'select',
        'options': ['Two-Door Car', 'Four-Door Car', 'SUV', 'Luxury Car', 'Sports Car']
    },
    'Coverage': {
        'type': 'select',
        'options': ['Basic', 'Extended', 'Premium']
    },
    'Renew Offer Type': {
        'type': 'select',
        'options': ['Offer1', 'Offer2', 'Offer3', 'Offer4']
    },
    'EmploymentStatus': {
        'type': 'select', 
        'options': ['Employed', 'Unemployed', 'Medical Leave', 'Disabled', 'Retired']
    },
    'Marital Status': {
        'type': 'select',
        'options': ['Married', 'Single', 'Divorced']
    },
    'Education': {
        'type': 'select',
        'options': ['High School', 'Bachelor', 'Master', 'Doctor']
    },
    'Number of Policies': {
        'type': 'number',
        'min': 1,
        'max': 10,
        'value': 2
    },
    'Monthly Premium Auto': {
        'type': 'number',
        'min': 50,
        'max': 300,
        'value': 100
    },
    'Total Claim Amount': {
        'type': 'number',
        'min': 0,
        'max': 1000,
        'value': 400
    },
    'Income': {
        'type': 'number',
        'min': 0,
        'max': 1000000,
        'value': 50000
    }
}

# Fungsi membuat input form
def create_input_form():
    inputs = {}
    cols = st.columns(2)
    
    for i, (feature, config) in enumerate(FEATURES.items()):
        with cols[i % 2]:
            if config['type'] == 'select':
                inputs[feature] = st.selectbox(
                    label=feature,
                    options=config['options'],
                    key=feature
                )
            else:
                inputs[feature] = st.number_input(
                    label=feature,
                    min_value=config['min'],
                    max_value=config['max'],
                    value=config['value'],
                    key=feature
                )
    return inputs

# Form input manual
st.header('Input Data Pelanggan')
with st.form("input_form"):
    input_data = create_input_form()
    submitted = st.form_submit_button("Lakukan Prediksi")

# Jika model berhasil dimuat dan form disubmit
if model and submitted:
    try:
        # Konversi input ke DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Lakukan prediksi
        prediction = model.predict(input_df)
        
        # Tampilkan hasil
        st.success('### Hasil Prediksi')
        st.metric(label="Predicted Value", value=f"{prediction[0]:.2f}")
        
        # Tampilkan data input
        st.subheader("Detail Input")
        st.dataframe(input_df.T.rename(columns={0: 'Value'}))
        
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {str(e)}")

# Panduan penggunaan
st.sidebar.header('Petunjuk Penggunaan')
st.sidebar.markdown("""
1. Isi semua field input sesuai dengan data pelanggan
2. Klik tombol **Lakukan Prediksi**
3. Hasil prediksi akan muncul di bagian bawah
4. Pastikan input sesuai dengan ketentuan:
   - **Vehicle Class**: Jenis kendaraan
   - **Coverage**: Tingkat perlindungan
   - **Renew Offer Type**: Tipe penawaran perpanjangan
   - **EmploymentStatus**: Status pekerjaan
   - **Education**: Tingkat pendidikan
""")

# Catatan model
st.sidebar.header('Informasi Model')
st.sidebar.markdown("""
- Menggunakan Random Forest Classifier
- Preprocessing otomatis untuk data kategorikal
- Model telah melalui optimasi hyperparameter
""")