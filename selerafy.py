import streamlit as st
import pandas as pd
import numpy as np
import spotipy
import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.cache_handler import MemoryCacheHandler

import hdbscan
import umap


# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="Selerafy Rock – HDBSCAN", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #191414; }
h1, h2, h3, h4, h5, h6, p, div, span { color: white !important; }
[data-testid="stSidebar"] { background-color: #000000; }
div.stButton > button:first-child {
    background-color: #1DB954;
    color: white;
    border-radius: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ======================================================
# SPOTIFY API
# ======================================================
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
except:
    HAS_SPOTIFY = False


# ======================================================
# LOAD DATA
# ======================================================
@st.cache_resource
def load_data():
    df = pd.read_csv("rock_track.csv")

    # only lowercase column names (NOT content)
    df.columns = [c.strip().lower() for c in df.columns]

    feature_cols = [
        "danceability", "energy", "key", "loudness", "mode",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo"
    ]

    df = df.dropna(subset=feature_cols)
    df = df.drop_duplicates(subset=["track", "artist"])
    df = df.reset_index(drop=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])

    return df, scaled, feature_cols

df, scaled_features, feature_cols = load_data()


# ======================================================
# HDBSCAN CLUSTERING
# ======================================================
@st.cache_resource
def run_hdbscan(X):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="leaf"
    )
    labels = clusterer.fit_predict(X)
    probabilities = clusterer.probabilities_
    return labels, probabilities

labels, probs = run_hdbscan(scaled_features)
df["cluster"] = labels
df["cluster_prob"] = probs


# ======================================================
# UMAP VISUALIZATION
# ======================================================
@st.cache_resource
def compute_umap(X):
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )
    return reducer.fit_transform(X)

umap_embeddings = compute_umap(scaled_features)
df["umap_x"] = umap_embeddings[:, 0]
df["umap_y"] = umap_embeddings[:, 1]


# ======================================================
# RECOMMENDER
# ======================================================
def recommend(song, artist, top_n=5):
    # ❗ FIX: comparing artist EXACT (no lowercase)
    mask = (df["track"] == song) & (df["artist"] == artist)

    if mask.sum() == 0:
        return None, None

    idx = df[mask].index[0]
    cluster_id = df.loc[idx, "cluster"]
    target_vec = scaled_features[idx].reshape(1, -1)

    # Cluster exists
    if cluster_id != -1:
        same_cluster = df[df["cluster"] == cluster_id]
        same_idx = same_cluster.index

        dist_local = euclidean_distances(
            target_vec,
            scaled_features[same_idx]
        ).flatten()

        sorted_local = np.argsort(dist_local)[1:top_n+1]
        rec_local = same_cluster.iloc[sorted_local]
        return rec_local, idx

    # Outlier fallback
    dist_global = euclidean_distances(target_vec, scaled_features).flatten()
    sorted_global = np.argsort(dist_global)[1:top_n+1]
    rec_global = df.iloc[sorted_global]
    return rec_global, idx


# ======================================================
# RADAR CHART
# ======================================================
def plot_radar(track_idx, rec_indices):
    fig = go.Figure()

    base_vals = scaled_features[track_idx].tolist()
    base_vals.append(base_vals[0])

    fig.add_trace(go.Scatterpolar(
        r=base_vals,
        theta=feature_cols + [feature_cols[0]],
        fill="toself",
        name="Lagu Pilihan",
        line_color="#1DB954",
        fillcolor="rgba(29,185,84,0.3)"
    ))

    rec_vals = np.mean(scaled_features[rec_indices], axis=0).tolist()
    rec_vals.append(rec_vals[0])

    fig.add_trace(go.Scatterpolar(
        r=rec_vals,
        theta=feature_cols + [feature_cols[0]],
        fill="toself",
        name="Rekomendasi",
        line_color="#9B59B6",
        fillcolor="rgba(155,89,182,0.3)"
    ))

    fig.update_layout(polar=dict(radialaxis=dict(range=[-2, 3])))
    return fig


# ======================================================
# UI
# ======================================================
st.title("🎸 Selerafy Rock Edition – HDBSCAN")
st.caption("Clustering otomatis tanpa slider dengan HDBSCAN + UMAP")

df["display"] = df["track"] + " - " + df["artist"]
song_name = st.selectbox("Pilih lagu rock favoritmu:", df["display"])

if st.button("Cari Lagu Mirip 🔍"):
    row = df[df["display"] == song_name].iloc[0]

    recs, original_idx = recommend(row["track"], row["artist"])

    if recs is not None:
        st.success("✔ Rekomendasi berdasarkan struktur alami dataset (HDBSCAN).")

        col1, col2 = st.columns([1, 1])

        # === RECOMMENDATION LIST ===
        with col1:
            st.subheader("Hasil Rekomendasi")

            for _, r in recs.iterrows():
                with st.expander(f"{r['track']} – {r['artist']}"):
                    st.write(f"Cluster: {r['cluster']} | Prob: {r['cluster_prob']:.2f}")

                    if HAS_SPOTIFY:
                        try:
                            q = f"track:{r['track']} artist:{r['artist']}"
                            res = sp.search(q=q, type="track", limit=1)
                            if res["tracks"]["items"]:
                                item = res["tracks"]["items"][0]
                                st.image(item["album"]["images"][0]["url"], width=120)
                                if item["preview_url"]:
                                    st.audio(item["preview_url"])
                                st.markdown(f"[🔗 Buka di Spotify]({item['external_urls']['spotify']})")
                        except:
                            pass

        # === RADAR CHART ===
        with col2:
            st.subheader("Analisis Audio")
            fig = plot_radar(original_idx, recs.index)
            st.plotly_chart(fig, use_container_width=True)

        # === CLUSTER VISUALIZATION ===
        st.subheader("🎨 Visualisasi Cluster Musik Rock")
        fig2 = px.scatter(
            df,
            x="umap_x", y="umap_y",
            color=df["cluster"].astype(str),
            hover_data=["track", "artist"],
            title="Peta Musik Rock (UMAP + HDBSCAN)"
        )
        st.plotly_chart(fig2, use_container_width=True)
