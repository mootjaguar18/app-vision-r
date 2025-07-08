import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont
import firebase_admin
from firebase_admin import credentials, firestore
from ultralytics import YOLO
from collections import Counter

# ------------------------
# Inicializar Firebase
# ------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("vision-app-firebase-key.json")
    firebase_admin.initialize_app(cred)


db = firestore.client()

# ------------------------
# Cargar modelo YOLOv8
# ------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ------------------------
# Diccionario inglés -> español
# ------------------------
translation_dict = {
    "person": "persona",
    "car": "auto",
    "bicycle": "bicicleta",
    "dog": "perro",
    "cat": "gato",
    "bus": "camión",
    "truck": "camión pesado",
    "motorcycle": "motocicleta",
    "traffic light": "semáforo",
    "stop sign": "señal de alto",
    # Añadir más si lo deseas
}

# ------------------------
# Función para colores dominantes
# ------------------------
def get_dominant_colors(image_bgr, k=5):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    total = np.sum(counts)
    percentages = [(count / total) * 100 for count in counts]
    sorted_indices = np.argsort(percentages)[::-1]
    return colors[sorted_indices], [percentages[i] for i in sorted_indices]

# ------------------------
# Conversión a HEX
# ------------------------
def rgb_to_hex(color):
    return '#{:02x}{:02x}{:02x}'.format(*color)

# ------------------------
# Detección de objetos
# ------------------------
def detect_objects(image_rgb):
    results = model(image_rgb)
    boxes = results[0].boxes
    names = results[0].names

    detections = []
    coords = []

    for box in boxes:
        cls_id = int(box.cls[0])
        label_en = names[cls_id]
        label_es = translation_dict.get(label_en, label_en)
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        detections.append(label_es)
        coords.append((xyxy, label_es, conf))

    return detections, coords

# ------------------------
# Dibujar recuadros con texto (sin textsize)
# ------------------------
def draw_boxes(image_pil, coords):
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()

    for (xyxy, label, conf) in coords:
        x1, y1, x2, y2 = xyxy
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        text = f"{label} {conf:.2f}"

        # Estimar tamaño del texto
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # Compatibilidad con Pillow viejo
            text_width, text_height = font.getsize(text)

        # Fondo para el texto
        draw.rectangle(
            [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
            fill="green"
        )
        draw.text((x1 + 2, y1 - text_height - 2), text, fill="white", font=font)

    return image_pil

# ------------------------
# App principal
# ------------------------
st.set_page_config(page_title="Clasificación de Imagen", layout="wide")
st.title("Detección de Colores y Objetos con Firebase")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Imagen Original", use_column_width=True)

    # Colores dominantes
    colors, percentages = get_dominant_colors(image_bgr)
    st.subheader("Colores dominantes:")
    hex_colors = []

    for color, percent in zip(colors, percentages):
        hex_color = rgb_to_hex(color)
        hex_colors.append({"color": hex_color, "percentage": round(percent, 2)})
        st.markdown(
            f"<div style='display:flex; align-items:center;'>"
            f"<div style='width:40px; height:25px; background-color:{hex_color}; margin-right:10px; border:1px solid black;'></div>"
            f"<span>{hex_color} - {percent:.2f}%</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    # Detección de objetos
    detections, coords = detect_objects(image_np)
    counts = Counter(detections)

    st.subheader("Objetos detectados:")
    if counts:
        for label, count in counts.items():
            st.write(f"- {label}: {count}")
    else:
        st.info("No se detectaron objetos en la imagen.")

    # Mostrar imagen anotada
    image_with_boxes = draw_boxes(image.copy(), coords)
    st.image(image_with_boxes, caption="Imagen con detecciones", use_column_width=True)

    # Guardar en Firestore
    doc = {
        "colores": hex_colors,
        "objetos": dict(counts),
        "nombre_imagen": uploaded_file.name,
        "usuario": "usuario_demo"
    }

    db.collection("detecciones").add(doc)
    st.success("Datos guardados correctamente en Firebase Firestore.")
