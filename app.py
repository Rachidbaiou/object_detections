import cv2
import numpy as np
import torch
from ultralytics import YOLO
import io
from PIL import Image
import os
import streamlit as st

# Charger le modèle YOLO
model = YOLO('yolov9c.pt')

# Chemin du dossier temporaire pour stocker les images téléchargées
UPLOAD_FOLDER = 'uploads'

# Vérifiez si le dossier existe, sinon, créez-le
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

st.title('Detection de bouteilles dans un pack :')

# Afficher le formulaire de téléchargement de fichier
uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("Prendre une photo", key="camera", label_visibility="hidden")

def process_image(image):
    # Convertir l'image en tableau numpy
    img_np = np.array(image)
    
    # Détecter les objets dans l'image
    results = model(img_np)
    class_names = model.names
    
    # Classe cible (bouteille)
    target_class = 'bottle'
    target_class_id = None

    # Trouver l'ID de la classe cible
    for idx, class_name in enumerate(class_names):
        if class_name == target_class:
            target_class_id = idx
            break

    if target_class_id is None:
        st.write("Aucune bouteille n'est trouvee dans l'image.")
        return img_np, 0, {}

    # Initialiser le compteur d'objets
    total_objects = 0
    
    # Compteur pour chaque type d'objet détecté
    object_counts = {}

    # Dessiner les résultats sur l'image
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Les coordonnées des bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Les scores de confiance
        classes = result.boxes.cls.cpu().numpy()  # Les classes prédites
        
        for box, score, cls in zip(boxes, scores, classes):
            if int(cls) == target_class_id:
                cv2.putText(int(cls))
                total_objects += 1
                x1, y1, x2, y2 = map(int, box)  # Convertir les coordonnées en entiers
                label = f"{class_names[int(cls)]}: {score:.2f}"  # Créer le label à afficher
                
                # Dessiner le rectangle
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Dessiner le label
                cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Compter le nombre d'occurrences de chaque type d'objet détecté
                class_name = class_names[int(cls)]
                object_counts[class_name] = object_counts.get(class_name, 0) + 1

    # Ajouter le nombre total d'objets détectés sur l'image
    total_objects_label = f"Total Bottles Detected: {total_objects}"
    cv2.putText(img_np, total_objects_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return img_np, total_objects, object_counts

if uploaded_file is not None:
    # Charger l'image depuis le fichier uploadé
    img = Image.open(uploaded_file)
    processed_img, total_objects, object_counts = process_image(img)
    
    # Afficher l'image annotée
    st.image(processed_img, channels="BGR")
    # Afficher le nombre total d'objets détectés et les types d'objets
    st.write(f"Nombre total de bouteilles détectées : {total_objects}")

    # Afficher le nombre d'occurrences de chaque type d'objet détecté
   

if camera_file is not None:
    # Charger l'image depuis la caméra
    img = Image.open(camera_file)
    processed_img, total_objects, object_counts = process_image(img)
    
    # Afficher l'image annotée
    st.image(processed_img, channels="BGR")
    # Afficher le nombre total d'objets détectés et les types d'objets
    st.write(f"Nombre total de bouteilles détectées : {total_objects}")

    # Afficher le nombre d'occurrences de chaque type d'objet détecté
   
