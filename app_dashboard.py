import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="PFE - Détection Salissure PV", layout="wide")
st.title("☀️ Dashboard d'Inspection Intelligente - Panneaux PV")
st.write("Projet de Master en Informatique Industrielle - ISI Médenine")

# --- CHARGEMENT DU MODÈLE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, r'C:\Users\INFOKOM ADMINN\Desktop\MON_PROJET_PFE\solar_dust_model_v1.keras')

@st.cache_resource # Pour ne charger le modèle qu'une seule fois
def load_my_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_my_model()

# --- INTERFACE ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload d'image")
    uploaded_file = st.file_uploader("Choisissez une photo du panneau...", type=["jpg", "jpeg", "png"])

with col2:
    st.header("🔍 Analyse AI")
    if uploaded_file is not None:
        # Affichage de l'image
        img = Image.open(uploaded_file)
        st.image(img, caption="Image sélectionnée", use_container_width=True)
        
        if model is not None:
            # Prétraitement (MobileNetV2)
            img_resized = img.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_batch = np.expand_dims(img_array, axis=0)
            img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
            
            # Prédiction
            prediction = model.predict(img_preprocessed, verbose=0)
            score = prediction[0][0]
            
            # Résultats
            if score > 0.5:
                st.error(f"RÉSULTAT : SALE (Poussière détectée)")
                st.metric("Confiance", f"{score*100:.2f}%")
                st.warning("⚠️ Action requise : Planifier un nettoyage.")
            else:
                st.success(f"RÉSULTAT : PROPRE")
                st.metric("Confiance", f"{(1-score)*100:.2f}%")
                st.info("✅ État optimal de production.")
        else:
            st.error("Erreur : Le fichier 'solar_pfe_model.keras' n'a pas été trouvé.")
    else:
        st.info("Veuillez uploader une image pour lancer l'analyse.")

st.divider()
st.caption("Développé par Ali Nouh - Expertise en Industrial Informatics")