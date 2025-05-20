#################################
# ⚙️ DATASET TEMPERATURES FUTURES    ⚙️#
#################################

from PIL import Image
import streamlit as st
from utils.assets_loader import load_image

def drias():

    #base_dir = os.path.dirname(os.path.abspath(__file__))  # chemin absolu du fichier divers_processing.py
    #image_path = os.path.join(base_dir, '..', 'pictures', 'Carto_safran.png')

    #if not os.path.exists(image_path):
     #   raise FileNotFoundError(f"Image introuvable à ce chemin : {image_path}")
    
    #Carto_safran = Image.open("../pictures/Carto_safran.png")
    #scenarios_climatiques_regions_france = Image.open("../pictures/scenarios_climatiques_regions_france.png")

    st.subheader("🌡️ Données Climatiques Futures pour la Simulation")

    st.markdown("""
    Pour tester notre modèle avec des **données futures réalistes**, nous devons récupérer une simulation des **températures régionales**, 
    à la **maille fine** et sur **plusieurs années**. Après exploration, nous avons identifié sur le portail de la [**DRIAS**](https://www.drias-climat.fr) un catalogue de données de **simulations climatiques** 
    au format [**NetCDF**] (https://fr.wikipedia.org/wiki/NetCDF) pouvant répondre à ce besoin.

    Choix d’un Scénario d’émission de Co2. Au format NetCDF

    - **RCP2.6** : Scénario avec une politique climatique visant à faire baisser les concentrations en CO2
    - **RCP4.5** : Scénario avec une politique climatique visant à stabiliser les concentrations en CO2
    - **RCP8.5** : Scénario sans politique climatique

    ---
    """)
    img = load_image("Carto_safran.png")
    if img:
            st.image(img, caption="Carte des données SAFRAN", use_column_width=True)
    else:
            st.warning("❌ L’image 'Carto_safran.png' est introuvable dans le dossier `pictures/`.")

    #st.image(scenarios_climatiques_regions_france, caption="Simulation des températures régionales", use_column_width=True)
    
    st.markdown("### 🎯 Hypothèses et Méthodologie retenues")
    
    st.markdown("""
    - 🔹 **Un seul scénario climatique** a été retenu, correspondant à une **hypothèse intermédiaire d’émissions de CO₂**.  
    - ⚠️ Les **prévisions climatiques** sont issues de **modèles complexes** qui comportent des **incertitudes** importantes.  
    - 📅 Les **données consolidées datent de 2020**, ce qui peut introduire un **biais temporel**.  
    - 🗺️ Pour associer les données climatiques à nos **régions administratives**, nous avons utilisé une méthode simple de **regroupement par encadrement** 
      des **coordonnées géographiques**.

    ---
    """)

    st.info("Cette étape est cruciale pour nourrir notre simulateur avec des températures cohérentes et projeter la consommation régionale future.")
