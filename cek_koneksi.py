import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

st.title("Tes Koneksi Spotify")

try:
    # 1. Baca Secrets
    cid = st.secrets["spotify"]["CLIENT_ID"]
    csecret = st.secrets["spotify"]["CLIENT_SECRET"]
    st.write(f"Client ID terbaca: {cid[:5]}... (depan saja)")
    
    # 2. Coba Login
    auth_manager = SpotifyClientCredentials(client_id=cid, client_secret=csecret)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    
    # 3. Coba Ambil 1 Lagu (Shape of You)
    track = sp.track("7qiZfU4dY1lWllzX7mPBI3")
    st.success(f"✅ BERHASIL! Terkoneksi ke Spotify. Lagu: {track['name']}")
    
except Exception as e:
    st.error(f"❌ GAGAL. Error: {e}")
    st.info("Jika error 403/401: Client ID/Secret salah atau belum di-add user.")
    st.info("Jika error FileNotFoundError: secrets.toml salah tempat.")