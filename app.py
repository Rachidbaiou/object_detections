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

st.title('YOLO Object Detection')

# Afficher le formulaire de téléchargement de fichier
uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Charger l'image depuis le fichier uploadé
    img = Image.open(uploaded_file)
    
    # Convertir l'image en tableau numpy
    img_np = np.array(img)
    
    # Détecter les objets dans l'image
    results = model(img_np)
    class_names = model.names
    
    # Initialiser le compteur d'objets
    total_objects = 0
    
    # Dessiner les résultats sur l'image
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Les coordonnées des bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Les scores de confiance
        classes = result.boxes.cls.cpu().numpy()  # Les classes prédites
        
        total_objects += len(boxes)  # Compter les objets détectés
        
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)  # Convertir les coordonnées en entiers
            label = f"{class_names[int(cls)]}: {score:.2f}"  # Créer le label à afficher
            
            # Dessiner le rectangle
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Dessiner le label
            cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Ajouter le nombre total d'objets détectés sur l'image
    total_objects_label = f"Total Objects Detected: {total_objects}"
    cv2.putText(img_np, total_objects_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Afficher l'image annotée
    st.image(img_np, channels="BGR")
