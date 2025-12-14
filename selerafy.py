import streamlit as st
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import re
import matplotlib.pyplot as plt
import seaborn as sns
import random

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Selerafy (Pencocokan Lagu Spotify)", page_icon="🔥", layout="centered")

# --- 1. SETUP SPOTIFY API ---
try:
    CLIENT_ID = st.secrets["spotify"]["CLIENT_ID"]
    CLIENT_SECRET = st.secrets["spotify"]["CLIENT_SECRET"]
except Exception:
    st.error("❌ File secrets.toml belum disetting.")
    st.stop()

# --- 2. FUNGSI MEMBERSIHKAN NAMA LAGU ---
def clean_track_name(name):
    if not isinstance(name, str): return ""
    name = name.lower()
    name = re.sub(r'\([^)]*\)', '', name) # Hapus (...)
    name = re.sub(r'\[[^]]*\]', '', name) # Hapus [...]
    name = name.replace('feat.', '').replace('remastered', '').replace('remix', '').replace('-', '')
    return " ".join(name.split())

# --- 3. LOAD DATABASE (DIPERBAIKI: Menambah Fitur Penting) ---
@st.cache_resource
def load_database_monster():
    try:
        # Pastikan file CSV ada di folder yang sama
        df = pd.read_csv('dataset_spotify_tracks.csv')
        df.columns = [c.lower() for c in df.columns]
        
        name_col = 'name' if 'name' in df.columns else 'track_name'
        if name_col not in df.columns:
            st.error("❌ Kolom nama lagu tidak ditemukan.")
            return None, None

        # FITUR YANG LEBIH LENGKAP UNTUK AKURASI
        required_features = [
            'danceability', 'energy', 'valence', 'acousticness', 
            'loudness', 'tempo', 'instrumentalness'
        ]
        
        # Buat Dictionary agar pencarian Cepat
        song_lookup = {}
        
        # Kita iterasi data
        for index, row in df.iterrows():
            if pd.notna(row[name_col]):
                clean_name = clean_track_name(row[name_col])
                if clean_name not in song_lookup:
                    # Simpan list fitur sesuai urutan required_features
                    feats = []
                    valid = True
                    for col in required_features:
                        if col in df.columns:
                            feats.append(row[col])
                        else:
                            valid = False
                    if valid:
                        song_lookup[clean_name] = feats
        
        return df, song_lookup, required_features

    except FileNotFoundError:
        st.error("⚠️ File 'dataset_spotify_tracks.csv' tidak ditemukan.")
        return None, None, None

# Load Database
with st.spinner("Sedang memuat database & mengindeks fitur audio..."):
    df_database, song_dict, feature_cols = load_database_monster()

# --- 4. ENGINE PERHITUNGAN BARU (COSINE + NORMALISASI) ---
def calculate_similarity(playlist_features, target_features):
    """
    Menghitung kemiripan menggunakan Cosine Similarity dengan Normalisasi.
    """
    # 1. Gabungkan data untuk normalisasi bareng
    # playlist_features adalah List of Lists, target_features adalah List tunggal
    all_data = playlist_features + [target_features]
    
    df_all = pd.DataFrame(all_data)
    
    # 2. Normalisasi (Skala 0-1)
    # Ini PENTING supaya Tempo (120) tidak memakan Energy (0.8)
    scaler = MinMaxScaler()
    df_normalized = scaler.fit_transform(df_all)
    
    # 3. Pisahkan kembali
    playlist_norm = df_normalized[:-1] # Semua baris kecuali terakhir
    target_norm = df_normalized[-1].reshape(1, -1) # Baris terakhir
    
    # 4. Hitung Cosine Similarity
    # Mengukur rata-rata kemiripan target terhadap setiap lagu di playlist
    similarities = cosine_similarity(playlist_norm, target_norm)
    mean_score = np.mean(similarities) # Ambil rata-ratanya
    
    return mean_score * 100 # Jadikan persen

# --- 5. FUNGSI GET PLAYLIST DARI SPOTIFY ---
def get_playlist_features(playlist_url, sp, lookup_dict):
    try:
        pid = playlist_url.split("/")[-1].split("?")[0]
        results = sp.playlist_tracks(pid, limit=50)
        
        found_data = []
        found_names = []
        missed_count = 0
        
        for item in results['items']:
            track = item.get('track')
            if track:
                real_name = track['name']
                search_key = clean_track_name(real_name)
                
                # Cari di Database Kaggle
                features = lookup_dict.get(search_key)
                
                if features:
                    found_data.append(features)
                    found_names.append(real_name)
                else:
                    missed_count += 1
        
        return found_data, found_names, missed_count

    except Exception as e:
        return [], [], 0

# --- 6. USER INTERFACE (UI) ---

st.title("🔥 Spotify Smart Matcher")
st.markdown("Algoritma: **Cosine Similarity + Normalisasi Data**")

if song_dict:
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    # Input Playlist
    url = st.text_input("1. Paste Link Playlist Sumber (Contoh: Playlist Rock/Pop kamu):")

    if url:
        # Hapus validasi aneh sebelumnya, ganti dengan validasi standar spotify
        if "spotify.com" not in url:
             st.error("Link tidak valid. Harap masukkan link Spotify.")
        else:
            p_features, p_names, p_missed = get_playlist_features(url, sp, song_dict)
            
            if len(p_features) > 0:
                st.success(f"✅ Playlist terbaca! {len(p_features)} lagu ditemukan di database.")
                with st.expander(f"Lihat {len(p_features)} lagu yang menjadi acuan:"):
                    st.write(p_names)
                
                st.write("---")
                
                # --- MULAI KODE VISUALISASI ---
                with st.expander("📊 Lihat Visualisasi 'Vibe' Playlist Kamu (Linear Regression)"):
                    st.info("Grafik ini menunjukkan pola fitur audio playlistmu (Hijau) dibanding lagu acak (Merah).")
                    
                    # 1. SIAPKAN DATA DUMMY UNTUK PLOT
                    # Positif (Playlist Kamu)
                    df_pos = pd.DataFrame(p_features, columns=feature_cols)
                    df_pos['target'] = 10
                    
                    # Negatif (Ambil Sampel Random dari Database Raksasa sebagai Noise)
                    all_values = list(song_dict.values())
                    # Ambil 300 lagu random, atau sejumlah database jika kurang dari 300
                    n_sample = min(300, len(all_values))
                    random_feats = random.sample(all_values, n_sample)
                    
                    df_neg = pd.DataFrame(random_feats, columns=feature_cols)
                    df_neg['target'] = 0
                    
                    # Gabung untuk Visualisasi
                    df_viz = pd.concat([df_pos, df_neg])
                    
                    # 2. PILIH FITUR
                    viz_col1, viz_col2 = st.columns([1, 3])
                    with viz_col1:
                        feature_to_plot = st.selectbox(
                            "Pilih Sumbu X:", 
                            ['energy', 'danceability', 'valence', 'loudness', 'acousticness', 'tempo'],
                            index=0
                        )
                    
                    # 3. BUAT PLOT
                    with viz_col2:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        
                        # Scatter Plot
                        sns.scatterplot(
                            data=df_viz, x=feature_to_plot, y='target', 
                            hue='target', palette={0: '#e74c3c', 10: '#2ecc71'}, 
                            ax=ax, s=50, alpha=0.6, legend=False
                        )
                        
                        # Hitung Garis Regresi (Hanya untuk fitur yang dipilih)
                        X_v = df_viz[[feature_to_plot]]
                        y_v = df_viz['target']
                        model_viz = LinearRegression()
                        model_viz.fit(X_v, y_v)
                        
                        # Gambar Garis
                        x_range = np.linspace(df_viz[feature_to_plot].min(), df_viz[feature_to_plot].max(), 100).reshape(-1, 1)
                        y_pred = model_viz.predict(x_range)
                        ax.plot(x_range, y_pred, color='blue', linewidth=2, linestyle='--')
                        
                        ax.set_yticks([]) # Hilangkan angka sumbu Y biar bersih
                        ax.set_ylabel("Kemiripan")
                        ax.set_title(f"Distribusi {feature_to_plot.capitalize()}")
                        
                        st.pyplot(fig)
                # --- SELESAI KODE VISUALISASI ---
                # Input Target
                st.subheader("2. Bandingkan dengan Lagu Target")
                track_url = st.text_input("Paste Link Lagu Target:")
                
                if track_url:
                    try:
                        tid = track_url.split("/")[-1].split("?")[0]
                        t_info = sp.track(tid)
                        t_name = t_info['name']
                        t_key = clean_track_name(t_name)
                        
                        # Cari fitur lagu target
                        t_feats = song_dict.get(t_key)
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(t_info['album']['images'][0]['url'])
                        
                        with col2:
                            st.subheader(t_name)
                            st.write(f"Artis: **{t_info['artists'][0]['name']}**")
                            
                            if t_feats:
                                # --- PROSES PERHITUNGAN BARU ---
                                final_score = calculate_similarity(p_features, t_feats)
                                
                                # Visualisasi Warna
                                if final_score >= 80:
                                    color = "#2ecc71" # Hijau
                                    msg = "MATCHING BANGET! (Vibe Serupa)"
                                elif final_score >= 50:
                                    color = "#f1c40f" # Kuning
                                    msg = "Lumayan Mirip..."
                                else:
                                    color = "#e74c3c" # Merah
                                    msg = "Jauh Banget (Beda Genre)"
                                
                                st.markdown(f"<h1 style='color:{color}'>{final_score:.1f}%</h1>", unsafe_allow_html=True)
                                st.progress(int(final_score))
                                st.caption(msg)
                                
                                # Debugging (Opsional: Matikan kalau sudah oke)
                                with st.expander("Lihat Data Mentah (Untuk Debug)"):
                                    st.write(f"Fitur Lagu Target: {t_feats}")
                                    st.write("Fitur yang dipakai: Danceability, Energy, Valence, Acoustic, Loudness, Tempo, Instrumental")
                            
                            else:
                                st.warning("⚠️ Lagu target ini tidak ada di database Kaggle 1.2M.")
                                st.info("Coba lagu yang lebih mainstream/lama.")
                                
                    except Exception as e:
                        st.error(f"Gagal mengambil data lagu: {e}")
            else:
                st.error("Gagal membaca isi playlist atau tidak ada lagu yang cocok di database.")