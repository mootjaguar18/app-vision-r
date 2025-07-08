import streamlit as st
import cv2
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Inicializar Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("vision-app-b5e69-firebase-adminsdk-fbsvc-ba517bfde1.json")  # ← Cambia esto
    firebase_admin.initialize_app(cred)
db = firestore.client()

st.title("🔍 Visión Artificial + Firebase")
 
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen original", use_column_width=True)

    # Convertir a formato OpenCV
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Detección de objetos
    bbox, label, conf = cv.detect_common_objects(img_bgr)
    output_image = draw_bbox(img_bgr, bbox, label, conf)

    # Mostrar imagen procesada
    st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), caption="Resultado", use_column_width=True)
    st.write("Objetos detectados:", label)

    # Guardar en Firebase
    doc_ref = db.collection("detecciones").document()
    doc_ref.set({
        "archivo": uploaded_file.name,
        "objetos": label,
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    st.success("✅ Datos guardados en Firebase")
