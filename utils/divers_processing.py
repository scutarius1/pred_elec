#################################
# ⚙️ DATASET TEMPERATURES FUTURES    ⚙️#
#################################

from PIL import Image
import streamlit as st
from utils.assets_loader import load_image

def drias():
    st.subheader("🌡️ Données Climatiques Futures pour la Simulation")

    st.markdown("""
    Pour tester notre modèle avec des **données futures réalistes**, nous devons récupérer une simulation des **températures régionales**, 
    à la **maille fine** et sur **plusieurs années**. Après exploration, nous avons identifié sur le portail de la [**DRIAS**](https://www.drias-climat.fr) un catalogue de données de **simulations climatiques** 
    au format [**NetCDF**](https://fr.wikipedia.org/wiki/NetCDF) pouvant répondre à ce besoin.

    Choix d’un Scénario d’émission de Co2 : 

    - RCP2.6 : Scénario avec une politique climatique visant à faire baisser les concentrations en CO2
    - **RCP4.5** : Scénario avec une politique climatique visant à stabiliser les concentrations en CO2 (Scénario retenu)
    - RCP8.5 : Scénario sans politique climatique
    """)

    ###### image ######
    img = load_image("scenarios_climatiques_regions_france.png")
    if img:
            st.image(img, caption="Modèles possibles", use_container_width=True)
    else:
            st.warning("❌ L’image est introuvable dans le dossier `pictures/`.")
    ##################

    ###### image ######
    img = load_image("Carto_safran.png")
    if img:
            st.image(img, caption="Carte des données grille SAFRAN - plus de 8000 points sur le territoire", use_container_width=True)
    else:
            st.warning("❌ L’image est introuvable dans le dossier `pictures/`.")
    ##################
    


    st.markdown("### ⚠️ Avertissement")
    
    st.markdown("""
    - ⚠️ Les **prévisions climatiques** sont issues de **modèles complexes** qui comportent des **incertitudes** importantes.  
    - 📅 Les **données consolidées datent de 2020**, ce qui peut introduire un **biais temporel**.  
    - 🗺️ Pour associer les données climatiques à nos **régions administratives**, nous avons utilisé une méthode simple de **regroupement par encadrement** 
      des **coordonnées géographiques**.

    ---
    """)

    st.info("Cet exercice de simulation des températures futures vise uniquement à illustrer des méthodes de traitement de données " \
    "et nous permettre de mobiliser notre modèle de prédiction et simuler de manière triviale la future consommation électrique.")
