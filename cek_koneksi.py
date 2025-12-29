import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

st.title("Tes Koneksi Spotify")

try:
    # 1. Baca Secrets
    cid = st.secrets["spotify"]["CLIENT_ID"]
    csecret = st.secrets["spotify"]["CLIENT_SECRET"]
    st.write(f"Client ID terbaca: {cid[:5]}... (depan saja)")
    
    # 2. Coba Login
    auth_manager = SpotifyClientCredentials(client_id=cid, client_secret=csecret)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    
    # 3. Baca CSV dan ambil 1 lagu
    df = pd.read_csv("rock_track.csv")
    sample_track = df.iloc[0]  # Ambil lagu pertama
    track_name = sample_track['Track']
    artist_name = sample_track['Artist']
    
    st.write(f"📀 Mencari lagu dari CSV: {track_name} - {artist_name}")
    
    # 4. Cari lagu di Spotify
    query = f"{track_name} {artist_name}"
    results = sp.search(q=query, type='track', limit=1)
    
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        st.success(f"✅ BERHASIL! Terkoneksi ke Spotify.")
        st.write(f"🎵 Lagu ditemukan: {track['name']}")
        st.write(f"🎤 Artist: {track['artists'][0]['name']}")
        st.write(f"💿 Album: {track['album']['name']}")
        st.write(f"🔗 URL: {track['external_urls']['spotify']}")
    else:
        st.warning("⚠️ Koneksi berhasil, tapi lagu tidak ditemukan di Spotify")
    
except Exception as e:
    st.error(f"❌ GAGAL. Error: {e}")
    st.info("Jika error 403/401: Client ID/Secret salah atau belum di-add user.")
    st.info("Jika error FileNotFoundError: secrets.toml salah tempat.")