import os
from PIL import Image
import streamlit as st

# 📍 Fonction pour construire le chemin absolu vers une image
def get_image_path(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, '..', 'pictures', filename)

# 💾 Fonction de chargement unique d'une image
@st.cache_resource
def load_image(filename):
    path = get_image_path(filename)
    if os.path.exists(path):
        return Image.open(path)
    else:
        st.warning(f"Image non trouvée : {filename}")
        return None


