import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Configuración de la página
st.set_page_config(
    page_title="ShopMind - Motor de Recomendaciones", page_icon="🛍️", layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@st.cache_resource
def cargar_datos():
    df_ratings = pd.read_csv(
        os.path.join(BASE_DIR, "data", "processed", "ratings_filtrado.csv")
    )
    df_productos = pd.read_csv(
        os.path.join(BASE_DIR, "data", "processed", "productos_descripciones.csv")
    )
    embeddings = np.load(
        os.path.join(BASE_DIR, "data", "processed", "embeddings_productos.npy")
    )

    with open(os.path.join(BASE_DIR, "data", "processed", "modelo_als.pkl"), "rb") as f:
        modelo_als = pickle.load(f)

    with open(os.path.join(BASE_DIR, "data", "processed", "mapeos.pkl"), "rb") as f:
        mapeos = pickle.load(f)

    return df_ratings, df_productos, embeddings, modelo_als, mapeos


df_ratings, df_productos, embeddings, modelo_als, mapeos = cargar_datos()

user_to_idx = mapeos["user_to_idx"]
product_to_idx = mapeos["product_to_idx"]
idx_to_product = product_to_idx

# Reconstruir matriz
df_ratings["u_idx"] = df_ratings["userId"].map(user_to_idx)
df_ratings["p_idx"] = df_ratings["productId"].map(
    {v: k for k, v in idx_to_product.items()}
)
df_ratings = df_ratings.dropna(subset=["u_idx", "p_idx"])
df_ratings["u_idx"] = df_ratings["u_idx"].astype(int)
df_ratings["p_idx"] = df_ratings["p_idx"].astype(int)

matriz = csr_matrix(
    (
        df_ratings["rating"].values,
        (df_ratings["u_idx"].values, df_ratings["p_idx"].values),
    ),
    shape=(len(user_to_idx), len(product_to_idx)),
)

# UI
st.title("🛍️ ShopMind — Motor de Recomendaciones")
st.markdown(
    "Sistema de recomendaciones de productos de belleza usando **Collaborative Filtering** y **Embeddings Semánticos**"
)

tab1, tab2 = st.tabs(
    ["👤 Por Usuario (Collaborative Filtering)", "🔍 Por Producto (Embeddings)"]
)

with tab1:
    st.subheader("Recomendaciones personalizadas por usuario")
    usuarios_disponibles = list(user_to_idx.keys())[:100]
    usuario = st.selectbox("Selecciona un usuario", usuarios_disponibles)
    n = st.slider("Número de recomendaciones", 5, 20, 10)

    if st.button("Recomendar", key="btn1"):
        u_idx = user_to_idx[usuario]
        user_items = matriz[u_idx]
        ids, scores = modelo_als.recommend(
            u_idx, user_items, N=n, filter_already_liked_items=True
        )

        st.markdown("### Productos recomendados:")
        for rank, (idx, score) in enumerate(zip(ids, scores), 1):
            producto_id = idx_to_product.get(idx, "Desconocido")
            desc = df_productos[df_productos["productId"] == producto_id][
                "descripcion"
            ].values
            descripcion = desc[0] if len(desc) > 0 else "Sin descripción"
            st.markdown(
                f"**{rank}. {producto_id}** — {descripcion} *(score: {score:.3f})*"
            )

with tab2:
    st.subheader("Productos similares por contenido")
    productos_disponibles = df_productos["productId"].tolist()[:200]
    producto = st.selectbox("Selecciona un producto", productos_disponibles)
    n2 = st.slider("Número de similares", 5, 20, 10, key="slider2")

    if st.button("Buscar similares", key="btn2"):
        idx = df_productos[df_productos["productId"] == producto].index[0]
        desc_consulta = df_productos.iloc[idx]["descripcion"]
        st.info(f"Descripción: {desc_consulta}")

        emb_consulta = embeddings[idx].reshape(1, -1)
        similitudes = cosine_similarity(emb_consulta, embeddings)[0]
        indices_similares = similitudes.argsort()[::-1][1 : n2 + 1]

        st.markdown("### Productos similares:")
        for rank, i in enumerate(indices_similares, 1):
            p = df_productos.iloc[i]
            score = similitudes[i]
            st.markdown(
                f"**{rank}. {p['productId']}** — {p['descripcion']} *(score: {score:.3f})*"
            )
