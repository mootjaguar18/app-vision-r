import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore

# Inicializar Firebase Admin con tu archivo JSON
if not firebase_admin._apps:
    cred = credentials.Certificate("vision-app-b5e69-firebase-adminsdk-fbsvc-ba517bfde1.json")
    firebase_admin.initialize_app(cred)

# Inicializa Firestore
db = firestore.client()

# Función para obtener colores dominantes
def get_dominant_colors(image, k=5):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    total = np.sum(counts)
    percentages = [(count / total) * 100 for count in counts]
    sorted_indices = np.argsort(percentages)[::-1]
    sorted_colors = colors[sorted_indices]
    sorted_percentages = [percentages[i] for i in sorted_indices]
    return sorted_colors, sorted_percentages

# Conversión RGB a HEX
def rgb_to_hex(color):
    return '#{:02x}{:02x}{:02x}'.format(*color)

# Interfaz de usuario
st.title("Detección de Colores Dominantes con Firebase")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Convertir imagen a formato BGR para OpenCV
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Imagen original", use_column_width=True)

    # Análisis de colores
    colors, percentages = get_dominant_colors(image_bgr, k=5)

    st.subheader("Colores dominantes:")
    hex_colors = []
    for color, percent in zip(colors, percentages):
        hex_color = rgb_to_hex(color)
        hex_colors.append({"color": hex_color, "percentage": round(percent, 2)})
        st.markdown(
            f"<div style='display:flex; align-items:center;'>"
            f"<div style='width:50px; height:25px; background-color:{hex_color}; margin-right:10px;'></div>"
            f"<span>{hex_color} - {percent:.2f}%</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    # Guardar resultados en Firebase Firestore
    st.success("Colores analizados. Guardando en Firebase...")
    doc_ref = db.collection("analisis_colores").add({
        "colores": hex_colors,
        "nombre_imagen": uploaded_file.name,
        "usuario": "usuario_demo",  # puedes usar autenticación más adelante
    })
    st.info("Datos guardados en Firebase correctamente.")
