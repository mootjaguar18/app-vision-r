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
# Inicialización Firebase
# ------------------------
def init_firebase(json_path: str):
    if not firebase_admin._apps:
        cred = credentials.Certificate(json_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firebase("vision-app-b5e69-firebase-adminsdk-fbsvc-ba517bfde1.json")

# ------------------------
# Carga modelo YOLOv8
# ------------------------
@st.cache_resource
def load_yolo_model(model_name="yolov8n.pt"):
    return YOLO(model_name)

model = load_yolo_model()

# ------------------------
# Diccionario traducción inglés -> español (ampliable)
# ------------------------
TRANSLATION_DICT = {
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
    # Añade más si lo necesitas
}

# ------------------------
# Función para obtener colores dominantes
# ------------------------
def get_dominant_colors(image_bgr: np.ndarray, k: int = 5):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    total = counts.sum()
    percentages = [(count / total) * 100 for count in counts]
    sorted_indices = np.argsort(percentages)[::-1]
    return colors[sorted_indices], [percentages[i] for i in sorted_indices]

# ------------------------
# Conversión RGB a HEX
# ------------------------
def rgb_to_hex(color: np.ndarray) -> str:
    return '#{:02x}{:02x}{:02x}'.format(*color)

# ------------------------
# Detectar objetos con YOLO
# ------------------------
def detect_objects(image_rgb: np.ndarray):
    results = model(image_rgb)
    boxes = results[0].boxes
    names = results[0].names

    detections = []
    coords = []

    for box in boxes:
        cls_id = int(box.cls[0])
        label_en = names[cls_id]
        label_es = TRANSLATION_DICT.get(label_en, label_en)
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        detections.append(label_es)
        coords.append((xyxy, label_es, conf))

    return detections, coords

# ------------------------
# Dibujar cajas verdes con texto sobre la imagen PIL
# ------------------------
def draw_boxes(image_pil: Image.Image, coords):
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()

    for (xyxy, label, conf) in coords:
        x1, y1, x2, y2 = xyxy
        # Dibujar rectángulo verde
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

        text = f"{label} {conf:.2f}"
        # Intentar obtener bbox texto (compatible con Pillow actual)
        try:
            bbox = draw.textbbox((x1, y1 - 15), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = font.getsize(text)

        # Fondo para texto (con margen)
        rect_start = (x1, y1 - text_height - 5)
        rect_end = (x1 + text_width + 4, y1)
        draw.rectangle([rect_start, rect_end], fill="green")
        draw.text((x1 + 2, y1 - text_height - 3), text, fill="white", font=font)

    return image_pil

# ------------------------
# App Streamlit
# ------------------------
def main():
    st.set_page_config(page_title="Análisis de Imagen - Colores y Objetos", layout="wide")
    st.title("Análisis de Colores Dominantes y Clasificación de Objetos")

    uploaded_file = st.file_uploader("Sube una imagen (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        st.image(image, caption="Imagen Original", use_column_width=True)

        # Colores dominantes
        colors, percentages = get_dominant_colors(image_bgr)
        st.subheader("Colores dominantes")
        hex_colors = []
        for color, perc in zip(colors, percentages):
            hex_col = rgb_to_hex(color)
            hex_colors.append({"color": hex_col, "percentage": round(perc, 2)})
            st.markdown(
                f"<div style='display:flex; align-items:center;'>"
                f"<div style='width:40px; height:25px; background-color:{hex_col}; margin-right:10px; border:1px solid #000;'></div>"
                f"<span>{hex_col} — {perc:.2f}%</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Detección objetos
        detections, coords = detect_objects(image_np)
        counts = Counter(detections)

        st.subheader("Objetos detectados y contados")
        if counts:
            for label, count in counts.items():
                st.write(f"• **{label.capitalize()}**: {count}")
        else:
            st.info("No se detectaron objetos en la imagen.")

        # Dibujar detección sobre la imagen
        image_annotated = draw_boxes(image.copy(), coords)
        st.image(image_annotated, caption="Imagen con objetos detectados", use_column_width=True)

        # Guardar en Firebase
        doc = {
            "colores": hex_colors,
            "objetos": dict(counts),
            "nombre_imagen": uploaded_file.name,
            "usuario": "usuario_demo",
        }
        db.collection("analisis_imagenes").add(doc)
        st.success("Datos guardados correctamente en Firebase.")

if __name__ == "__main__":
    main()
