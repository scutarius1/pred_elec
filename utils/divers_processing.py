#################################
# ⚙️ DATASET TEMPERATURES FUTURES    ⚙️#
#################################

from PIL import Image

def drias():

    Carto_safran = Image.open("../pictures/Carto_safran.png")
    scenarios_climatiques_regions_france = Image.open("../pictures/scenarios_climatiques_regions_france.png")

    st.subheader("🌡️ Données Climatiques Futures pour la Simulation")

    st.markdown("""
    Pour tester notre modèle avec des **données futures réalistes**, nous avons dû récupérer une simulation des **températures régionales**, 
    à la **maille fine** et sur **plusieurs années**.

    Après exploration, nous avons identifié sur le portail de la [**DRIAS**](https://www.drias-climat.fr) un catalogue de données de **simulations climatiques** 
    pouvant répondre à ce besoin.

    ---
    """)

    st.image(scenarios_climatiques_regions_france, caption="Simulation des températures régionales", use_column_width=True)
    
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
