import streamlit as st
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from spotipy.cache_handler import MemoryCacheHandler

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Rock Recommender", layout="wide")

# --- CUSTOM CSS (TEMA SPOTIFY) ---
st.markdown("""
<style>
    /* Mengubah background utama */
    .stApp {
        background-color: #191414;
    }
    
    /* Mengubah warna teks default */
    h1, h2, h3, h4, h5, h6, p, div, span {
        color: #FFFFFF !important;
    }
    
    /* Mengubah warna Sidebar */
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    
    /* Mengubah warna tombol (Primary Button) */
    div.stButton > button:first-child {
        background-color: #1DB954;
        color: white;
        border-radius: 20px;
        border: none;
        font-weight: bold;
    }
    
    /* Efek hover pada tombol */
    div.stButton > button:first-child:hover {
        background-color: #1ed760;
        color: white;
    }
    
    /* Mengubah warna input text */
    div[data-baseweb="input"] > div {
        background-color: #282828;
        color: white;
        border-color: #1DB954;
    }
    input {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ... Lanjutkan dengan kode load_data dan lainnya ...

# --- 1. SETUP SPOTIFY API (Opsional) ---
try:
    CLIENT_ID = st.secrets["spotify"]["CLIENT_ID"]
    CLIENT_SECRET = st.secrets["spotify"]["CLIENT_SECRET"]
    auth_manager = SpotifyClientCredentials(
        client_id=CLIENT_ID, 
        client_secret=CLIENT_SECRET,
        cache_handler=MemoryCacheHandler()
    )
    sp = spotipy.Spotify(auth_manager=auth_manager)
    HAS_SPOTIFY = True
except Exception:
    HAS_SPOTIFY = False

# --- 2. LOAD & PROCESS DATABASE ---
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('rock_track.csv')
        
        # 1. Ubah semua nama kolom jadi huruf kecil (Track -> track, Artist -> artist)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # 2. Cek apakah kolom 'track' dan 'artist' ada (Sesuai CSV Anda)
        if 'track' not in df.columns:
            st.error("❌ Kolom 'Track' tidak ditemukan di CSV.")
            return None, None, None
            
        # 3. Fitur Audio yang akan dipakai
        # Pastikan nama ini cocok dengan CSV (setelah di-lowercase)
        feature_cols = ['danceability', 'energy', 'valence', 'acousticness', 
                       'loudness', 'tempo', 'instrumentalness']
        
        # 4. Bersihkan data
        # Hapus baris yang kosong di kolom fitur
        df = df.dropna(subset=feature_cols)
        
        # Hapus duplikat lagu (berdasarkan judul dan artis)
        if 'artist' in df.columns:
            df = df.drop_duplicates(subset=['track', 'artist'])
        else:
            df = df.drop_duplicates(subset=['track'])
            
        df = df.reset_index(drop=True)
        
        # 5. Normalisasi Data (Penting untuk Cosine Similarity)
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df[feature_cols])
        
        return df, scaled_features, feature_cols

    except Exception as e:
        st.error(f"Error memuat data: {e}")
        return None, None, None

df, scaled_features, feature_cols = load_data()

# --- 3. FUNGSI REKOMENDASI ---
def get_recommendations(song_name, artist_name=None, top_n=5):
    # Cari index lagu yang dipilih user
    if artist_name and 'artist' in df.columns:
        mask = (df['track'] == song_name) & (df['artist'] == artist_name)
    else:
        mask = df['track'] == song_name
        
    if mask.sum() == 0:
        return None, None, None
        
    idx = df[mask].index[0]
    
    # Ambil vektor fitur lagu tersebut
    target_vector = scaled_features[idx].reshape(1, -1)
    
    # Hitung kemiripan dengan SEMUA lagu lain
    similarity_scores = cosine_similarity(target_vector, scaled_features).flatten()
    
    # Urutkan index dari skor tertinggi
    similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
    
    return df.iloc[similar_indices], similarity_scores[similar_indices], idx

# --- 4. VISUALISASI RADAR CHART ---
def plot_radar_chart(track_idx, rec_indices):
    categories = feature_cols
    
    fig = go.Figure()
    
    # Data Lagu Pilihan User
    user_vals = scaled_features[track_idx].tolist()
    user_vals.append(user_vals[0]) # Menutup loop
    
    fig.add_trace(go.Scatterpolar(
        r=user_vals,
        theta=categories + [categories[0]],
        fill='toself',
        name='Pilihanmu',
        line_color='blue'
    ))
    
    # Rata-rata Rekomendasi
    rec_vals = np.mean(scaled_features[rec_indices], axis=0).tolist()
    rec_vals.append(rec_vals[0])
    
    fig.add_trace(go.Scatterpolar(
        r=rec_vals,
        theta=categories + [categories[0]],
        fill='toself',
        name='Rekomendasi',
        line_color='orange',
        opacity=0.7
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# --- 5. USER INTERFACE ---
st.title("🎸 Selerafy Rock Edition")
st.caption("Unsupervised Learning: Cosine Similarity")

if df is not None:
    # Buat nama tampilan untuk Dropdown
    if 'artist' in df.columns:
        df['display_name'] = df['track'] + " - " + df['artist']
    else:
        df['display_name'] = df['track']
        
    song_list = df['display_name'].tolist()
    
    # Layout Input
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        selected_song_display = st.selectbox("Pilih / Ketik Lagu Rock Favoritmu:", song_list)
    
    with col_btn:
        st.write("") # Spacer
        st.write("") # Spacer
        search_clicked = st.button("Cari Lagu Mirip 🔍", type="primary")

    if search_clicked:
        # Ambil data asli dari pilihan dropdown
        row = df[df['display_name'] == selected_song_display].iloc[0]
        selected_track = row['track']
        selected_artist = row['artist'] if 'artist' in df.columns else None
        
        # Jalankan Rekomendasi
        recommendations, scores, original_idx = get_recommendations(selected_track, selected_artist)
        
        if recommendations is not None:
            st.success(f"Menemukan lagu yang mirip dengan **{selected_track}**!")
            
            # Bagi layar jadi 2 kolom: Kiri (Hasil), Kanan (Grafik)
            col_res, col_chart = st.columns([1, 1])
            
# ... (Bagian atas kode sama) ...
            
            with col_res:
                st.subheader("Daftar Lagu Mirip:")
                
                # Loop hasil rekomendasi
                for i, (index, row) in enumerate(recommendations.iterrows()):
                    score_pct = scores[i] * 100
                    track_name = row['track']
                    artist_name = row.get('artist', '')
                    
                    # --- LOGIKA LINK SPOTIFY ---
                    # 1. Cek apakah di CSV ada kolom 'track_id' atau 'id'
                    if 'track_id' in row:
                        spotify_url = f"https://open.spotify.com/track/{row['track_id']}"
                    elif 'id' in row:
                        spotify_url = f"https://open.spotify.com/track/{row['id']}"
                    else:
                        # 2. Jika tidak ada ID, buat Link Pencarian Otomatis
                        query = f"{track_name} {artist_name}".replace(" ", "%20")
                        spotify_url = f"https://open.spotify.com/search/{query}"

                    with st.expander(f"#{i+1} {track_name} ({score_pct:.1f}%)"):
                        st.write(f"Artis: **{artist_name}**")
                        
                        if HAS_SPOTIFY:
                            try:
                                q = f"track:{track_name} artist:{artist_name}"
                                res = sp.search(q=q, type='track', limit=1)
                                if res['tracks']['items']:
                                    item = res['tracks']['items'][0]
                                    st.image(item['album']['images'][0]['url'], width=100)
                                    if item['preview_url']:
                                        st.audio(item['preview_url'])
                            except:
                                pass
                            
                        # TOMBOL LINK
                        st.link_button("Putar di Spotify", spotify_url)
            
            with col_chart:
                st.subheader("Analisis Audio")
                fig = plot_radar_chart(original_idx, recommendations.index)
                st.plotly_chart(fig, use_container_width=True)
                st.info("Grafik ini membandingkan karakteristik audio (seperti tempo & energi) lagu pilihanmu dengan hasil rekomendasi.")
                
        else:
            st.error("Gagal menemukan lagu tersebut di database.")