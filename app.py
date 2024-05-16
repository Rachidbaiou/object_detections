import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Charger le modèle YOLO
model = YOLO('yolov5s.pt')

# Fonction pour traiter l'image
def process_image(img_np):
    # Convertir l'image de BGR à RGB
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # Détecter les objets dans l'image
    results = model(img_np)
    class_names = model.names

    # Initialiser le compteur d'objets
    total_objects = 0

    # Dessiner les résultats sur l'image
    for result in results.xyxy:
        for *box, conf, cls in result:
            x1, y1, x2, y2 = map(int, box)  # Convertir les coordonnées en entiers
            label = f"{class_names[int(cls)]}: {conf:.2f}"  # Créer le label à afficher

            # Dessiner le rectangle
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Dessiner le label
            cv2.putText(img_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            total_objects += 1

    # Ajouter le nombre total d'objets détectés sur l'image
    total_objects_label = f"Total Objects Detected: {total_objects}"
    cv2.putText(img_rgb, total_objects_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Afficher l'image annotée
    st.image(img_rgb, channels="RGB")

st.title('YOLO Object Detection')

# Afficher les options pour télécharger une image ou prendre une photo avec la caméra
option = st.radio("Choisissez une option:", ('Télécharger une image', 'Prendre une photo avec la caméra'))

if option == 'Télécharger une image':
    # Afficher le formulaire de téléchargement de fichier
    uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Charger l'image depuis le fichier uploadé
        img = Image.open(uploaded_file)
        img_np = np.array(img)
        process_image(img_np)

elif option == 'Prendre une photo avec la caméra':
    # Bouton pour prendre une photo avec la caméra
    if st.button('Prendre une photo'):
        # Utiliser st.camera_input() pour capturer une image depuis la caméra
        img_data = st.image('Prendre une photo', channels='BGR')
        img_np = np.array(img_data)
        process_image(img_np)
