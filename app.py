import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont
import firebase_admin
from firebase_admin import credentials, firestore
from ultralytics import YOLO
from collections import Counter

# --- Inicializar Firebase ---
if not firebase_admin._apps:
    cred = credentials.Certificate("vision-app-b5e69-firebase-adminsdk-fbsvc-ba517bfde1.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Carga modelo YOLOv8 pequeño ---
model = YOLO('yolov8n.pt')  # Cambia a otro modelo si quieres

# --- Funciones ---

import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont
import firebase_admin
from firebase_admin import credentials, firestore
from ultralytics import YOLO
from collections import Counter

# --- Inicializar Firebase ---
if not firebase_admin._apps:
    cred = credentials.Certificate("vision-app-b5e69-firebase-adminsdk-fbsvc-ba517bfde1.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Carga modelo YOLOv8 pequeño ---
model = YOLO('yolov8n.pt')  # Cambia a otro modelo si quieres

# Diccionario traducción inglés -> español
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
    # agrega más categorías si quieres
}

# --- Funciones ---

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
    sorted_colors = colors[sorted_indices]
    sorted_percentages = [percentages[i] for i in sorted_indices]
    return sorted_colors, sorted_percentages

def rgb_to_hex(color):
    return '#{:02x}{:02x}{:02x}'.format(*color)

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

def draw_boxes(image_pil, coords):
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()
    for (xyxy, label, conf) in coords:
        x1, y1, x2, y2 = xyxy
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        text = f"{label} {conf:.2f}"
        text_size = draw.textsize(text, font=font)
        draw.rectangle([x1, y1 - text_size[1], x1 + text_size[0], y1], fill="green")
        draw.text((x1, y1 - text_size[1]), text, fill="white", font=font)
    return image_pil

# --- Streamlit UI ---

st.title("Detección de Colores Dominantes y Clasificación de Objetos")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)

    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Imagen original", use_column_width=True)

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

    detections, coords = detect_objects(image_np)

    counts = Counter(detections)

    st.subheader("Objetos detectados y contados:")
    if counts:
        for label, count in counts.items():
            st.write(f"{label}: {count}")
    else:
        st.write("No se detectaron objetos.")

    image_with_boxes = draw_boxes(image.copy(), coords)
    st.image(image_with_boxes, caption="Imagen con detección de objetos", use_column_width=True)

    data_to_save = {
        "colores": hex_colors,
        "objetos": dict(counts),
        "nombre_imagen": uploaded_file.name,
        "usuario": "usuario_demo",
    }
    db.collection("analisis_imagenes").add(data_to_save)
    st.success("Datos guardados en Firebase correctamente.")


# --- Streamlit UI ---

st.title("Detección de Colores Dominantes y Clasificación de Objetos")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)

    # Imagen para OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Mostrar imagen original
    st.image(image, caption="Imagen original", use_column_width=True)

    # Detectar colores
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

    # Detectar objetos y obtener coordenadas
    detections, coords = detect_objects(image_np)

    # Contar objetos por categoría
    counts = Counter(detections)

    st.subheader("Objetos detectados y contados:")
    if counts:
        for label, count in counts.items():
            st.write(f"{label}: {count}")
    else:
        st.write("No se detectaron objetos.")

    # Dibujar cajas y etiquetas sobre la imagen
    image_with_boxes = draw_boxes(image.copy(), coords)
    st.image(image_with_boxes, caption="Imagen con detección de objetos", use_column_width=True)

    # Guardar datos en Firebase
    data_to_save = {
        "colores": hex_colors,
        "objetos": dict(counts),
        "nombre_imagen": uploaded_file.name,
        "usuario": "usuario_demo",
    }
    db.collection("analisis_imagenes").add(data_to_save)
    st.success("Datos guardados en Firebase correctamente.")
